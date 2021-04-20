import torch.nn as nn
import ltr.models.backbone as backbones
from ltr.models.neck import CorrNL,Depth_Corr,Naive_Corr
from ltr.models.head import bbox,corner_coarse,mask
from ltr import model_constructor


class SEcmnet(nn.Module):
    """ Scale Estimation network module with three branches: bbox, coner and mask. """
    def __init__(self, feature_extractor, neck_module, head_module, used_layers,
                 extractor_grad=True, unfreeze_layer3=False):
        """
        args:
            feature_extractor - backbone feature extractor
            bb_regressor - IoU prediction module
            bb_regressor_layer - List containing the name of the layers from feature_extractor, which are input to
                                    bb_regressor
            extractor_grad - Bool indicating whether backbone feature extractor requires gradients
        """
        super(SEcmnet, self).__init__()

        self.feature_extractor = feature_extractor
        self.neck = neck_module
        assert len(head_module) == 2
        self.corner_head, self.mask_head = head_module
        self.used_layers = used_layers

        self.feat_adjust = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ),

            nn.Sequential(
                nn.Conv2d(192, 256, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ),

            nn.Sequential(
                nn.Conv2d(480, 512, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(512),
                nn.ReLU()
            )
        ])

        if not extractor_grad:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, train_imgs, test_imgs, train_bb, mode='train'):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """
        self.forward_ref(train_imgs, train_bb)
        pred_dict = self.forward_test(test_imgs, mode)
        return pred_dict

    def forward_ref(self, train_imgs, train_bb):
        """ Forward pass of reference branch.
        size of train_imgs is (1,batch,3,H,W), train_bb is (1,batch,4)"""
        '''train_feat_dict's dtype is OrderedDict,key is 'layer3' '''
        num_sequences = train_imgs.shape[-4] # batch
        num_train_images = train_imgs.shape[0] if train_imgs.dim() == 5 else 1 # 1

        # Extract backbone features
        '''train_feat的数据类型都是OrderedDict,字典的键为'layer3' '''
        train_feats = self.extract_backbone_features(train_imgs.view(-1, *train_imgs.shape[-3:])) # 输入size是(batch,3,256,256)

        # get reference feature
        self.neck.get_ref_kernel([train_feats[-1]], train_bb.view(num_train_images, num_sequences, 4))

    def forward_test(self, test_imgs, mode='train'):
        """ Forward pass of test branch. size of test_imgs is (1,batch,3,256,256)"""
        output = {}
        # Extract backbone features
        '''test_feat_dict's dtype is OrderedDict,key is 'layer3' '''
        Lfeat_list = self.extract_backbone_features(test_imgs.view(-1, *test_imgs.shape[-3:]), layers=4)# 输入size是(batch,3,256,256)

        # fuse feature from two branches
        fusion_feat = self.neck.fuse_feat([Lfeat_list[-1]])
        # Obtain bbox prediction
        if mode=='train':
            output['corner'] = self.corner_head(fusion_feat)
            Lfeat_list = [self.feat_adjust[idx](feat) for idx, feat in enumerate(Lfeat_list[:3])]
            output['mask'] = self.mask_head(fusion_feat,Lfeat_list)
        elif mode=='test':
            output['feat'] = fusion_feat
            output['corner'] = self.corner_head(fusion_feat)
            Lfeat_list = [self.feat_adjust[idx](feat) for idx, feat in enumerate(Lfeat_list[:3])]
            output['mask'] = self.mask_head(fusion_feat, Lfeat_list)
        else:
            raise ValueError("mode should be train or test")
        return output

    def get_test_feat(self, test_imgs):
        """ Forward pass of test branch. size of test_imgs is (1,batch,3,256,256)"""
        # Extract backbone features
        '''test_feat_dict's dtype is OrderedDict,key is 'layer3' '''
        test_feat_dict = self.extract_backbone_features(test_imgs.view(-1, *test_imgs.shape[-3:]),
                                                        layers=['conv1','layer1','layer2','layer3'])# 输入size是(batch,3,256,256)
        '''list,其中每个元素对应一层输出的特征(tensor)'''
        # Save low-level feature list
        self.Lfeat_list = [feat for name, feat in test_feat_dict.items() if name != 'layer3']

        # fuse feature from two branches
        self.fusion_feat = self.neck.fuse_feat([test_feat_dict['layer3']])
        return self.fusion_feat

    def get_output(self,mode):
        if mode == 'corner':
            return self.corner_head(self.fusion_feat)
        elif mode == 'mask':
            return self.mask_head(self.fusion_feat, self.Lfeat_list)
        else:
            raise ValueError('mode should be bbox or corner or mask')

    def get_corner_heatmap(self):
        return self.corner_head.get_heatmap(self.fusion_feat)

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.used_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)

    def get_backbone_feat(self, test_imgs):
        """ Forward pass of test branch. size of test_imgs is (1,batch,3,256,256)"""
        # Extract backbone features
        '''test_feat_dict's dtype is OrderedDict,key is 'layer3' '''
        test_feat_dict = self.extract_backbone_features(test_imgs.view(-1, *test_imgs.shape[-3:]),
                                                        layers=['conv1','layer1','layer2','layer3'])  # 输入size是(batch,3,256,256)
        return test_feat_dict

    def compute_correlation(self, test_feat_dict):
        # fuse feature from two branches
        feat2 = [test_feat_dict['layer3']]
        if len(feat2) == 1:
            feat2 = feat2[0]
        '''Step1: pixel-wise correlation'''
        feat_corr,_ = self.neck.corr_fun(self.neck.ref_kernel, feat2)
        return feat_corr


@model_constructor
def SEcm_googlenet(backbone_pretrained=True,used_layers=('layer3',),pool_size=None,unfreeze_layer3=False,use_NL=True):
    # backbone
    backbone_net = backbones.googlenet(pretrained=backbone_pretrained)

    # neck
    neck_net = CorrNL.CorrNL(pool_size=pool_size, use_NL=use_NL)

    # multiple heads
    corner_head = corner_coarse.Corner_Predictor(inplanes=pool_size*pool_size)  # 64
    mask_head = mask.Mask_Predictor_fine()

    # net
    net = SEcmnet(feature_extractor=backbone_net,
                  neck_module=neck_net,
                  head_module=(corner_head, mask_head),
                  used_layers=used_layers, extractor_grad=True,
                  unfreeze_layer3=unfreeze_layer3)
    return net


