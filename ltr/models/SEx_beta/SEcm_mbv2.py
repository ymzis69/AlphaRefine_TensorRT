import torch.nn as nn
import ltr.models.backbone as backbones
from ltr.models.neck import CorrNL,Depth_Corr,Naive_Corr,CorrNL_mbv2_trt
from ltr.models.head import bbox,corner_coarse,mask
from ltr import model_constructor

import torch
from torch2trt import torch2trt

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
                nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ),

            nn.Sequential(
                nn.Conv2d(24, 256, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ),

            nn.Sequential(
                nn.Conv2d(32, 512, kernel_size=1, stride=1, padding=0),
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
        '''train_feat?????????????????????OrderedDict,???????????????'layer3' '''
        train_feats = self.extract_backbone_features(train_imgs.view(-1, *train_imgs.shape[-3:]), layers=4) # ??????size???(batch,3,256,256)

        # get reference feature
        self.neck.get_ref_kernel([train_feats[-1]], train_bb.view(num_train_images, num_sequences, 4))

    def forward_test(self, test_imgs, mode='train', convert_trt=False):
        """ Forward pass of test branch. size of test_imgs is (1,batch,3,256,256)"""
        output = {}
        if not convert_trt:
            # Extract backbone features
            '''test_feat_dict's dtype is OrderedDict,key is 'layer3' '''
            Lfeat_list = self.extract_backbone_features(test_imgs.view(-1, *test_imgs.shape[-3:]), layers=4)# ??????size???(batch,3,256,256)
        else:
            # tensorrt transform part 1
            backbone_mbv2_256_trt = torch2trt(self.extract_backbone_features, [test_imgs.view(-1, *test_imgs.shape[-3:])])
            Lfeat_list = backbone_mbv2_256_trt(test_imgs.view(-1, *test_imgs.shape[-3:]))
            torch.save(backbone_mbv2_256_trt.state_dict(), 'trt_models/trt_model_RF_RF_mbv2/backbone_mbv2_trt.pth')
            print('**********************TensorRT model 1 convert complete**********************')

        # fuse feature from two branches
        fusion_feat = self.neck.fuse_feat([Lfeat_list[-1]], convert_trt=convert_trt)
        # Obtain bbox prediction
        if mode=='train':
            output['corner'] = self.corner_head(fusion_feat)
            Lfeat_list = [self.feat_adjust[idx](feat) for idx, feat in enumerate(Lfeat_list[:3])]
            output['mask'] = self.mask_head(fusion_feat,Lfeat_list)
        elif mode=='test':
            output['feat'] = fusion_feat
            if not convert_trt:
                output['corner'] = self.corner_head(fusion_feat)
                Lfeat_list = [self.feat_adjust[idx](feat) for idx, feat in enumerate(Lfeat_list[:3])]
                output['mask'] = self.mask_head(fusion_feat, Lfeat_list[0], Lfeat_list[1], Lfeat_list[2])
            else:
                # tensorrt transform part 4
                corner_head_trt = torch2trt(self.corner_head, [fusion_feat])
                output['corner'] = corner_head_trt(fusion_feat)
                torch.save(corner_head_trt.state_dict(), 'trt_models/trt_model_RF_RF_mbv2/corner_head_trt.pth')
                print('**********************TensorRT model 4 convert complete**********************')
                # tensorrt transform part 5
                feat_adjust_0_trt = torch2trt(self.feat_adjust[0], [Lfeat_list[0]])
                Lfeat_list0 = feat_adjust_0_trt(Lfeat_list[0])
                torch.save(feat_adjust_0_trt.state_dict(), 'trt_models/trt_model_RF_RF_mbv2/Lfeat_list_trt0_trt.pth')

                feat_adjust_1_trt = torch2trt(self.feat_adjust[1], [Lfeat_list[1]])
                Lfeat_list1 = feat_adjust_1_trt(Lfeat_list[1])
                torch.save(feat_adjust_1_trt.state_dict(), 'trt_models/trt_model_RF_RF_mbv2/Lfeat_list_trt1_trt.pth')

                feat_adjust_2_trt = torch2trt(self.feat_adjust[2], [Lfeat_list[2]])
                Lfeat_list2 = feat_adjust_2_trt(Lfeat_list[2])
                torch.save(feat_adjust_2_trt.state_dict(), 'trt_models/trt_model_RF_RF_mbv2/Lfeat_list_trt2_trt.pth')
                print('**********************TensorRT model 5 convert complete**********************')
                # tensorrt transform part 6
                mask_head_trt = torch2trt(self.mask_head, [fusion_feat, Lfeat_list0, Lfeat_list1, Lfeat_list2])
                output['mask'] = mask_head_trt(fusion_feat, Lfeat_list0, Lfeat_list1, Lfeat_list2)
                torch.save(mask_head_trt.state_dict(), 'trt_models/trt_model_RF_RF_mbv2/mask_head_trt.pth')
                print('**********************TensorRT model 6 convert complete**********************')
                print('**********************mobilenetv2 model convert complete**********************')
        else:
            raise ValueError("mode should be train or test")
        return output

    def get_test_feat(self, test_imgs):
        """ Forward pass of test branch. size of test_imgs is (1,batch,3,256,256)"""
        # Extract backbone features
        '''test_feat_dict's dtype is OrderedDict,key is 'layer3' '''
        test_feat_dict = self.extract_backbone_features(test_imgs.view(-1, *test_imgs.shape[-3:]),
                                                        layers=['conv1','layer1','layer2','layer3'])# ??????size???(batch,3,256,256)
        '''list,?????????????????????????????????????????????(tensor)'''
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
            layers = 4
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)

    def get_backbone_feat(self, test_imgs):
        """ Forward pass of test branch. size of test_imgs is (1,batch,3,256,256)"""
        # Extract backbone features
        '''test_feat_dict's dtype is OrderedDict,key is 'layer3' '''
        test_feat_dict = self.extract_backbone_features(test_imgs.view(-1, *test_imgs.shape[-3:]),
                                                        layers=['conv1','layer1','layer2','layer3'])  # ??????size???(batch,3,256,256)
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
def SEcm_mobilenetv2(backbone_pretrained=True,used_layers=('layer3',),pool_size=None,unfreeze_layer3=False,use_NL=True):
    # backbone
    backbone_net = backbones.mobilenet_v2(pretrained=backbone_pretrained)
    # backbone_net = backbones.resnet18(pretrained=backbone_pretrained)

    # neck
    neck_net = CorrNL_mbv2_trt.CorrNL(pool_size=pool_size, use_NL=use_NL)

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


