from ltr.models.neck.neck_utils import *
import torch.nn as nn
#from thop import profile
#from thop import clever_format
import torch
#from torch.autograd import Variable

from torch2trt import torch2trt

class PtCorr(nn.Module):
    """Network module for IoU prediction. Refer to the ATOM paper for an illustration of the architecture.
    It uses two backbone feature layers as input.
    args:
        input_dim:  Feature dimensionality of the two input backbone layers.
        pred_input_dim:  Dimensionality input the the prediction network.
        pred_inter_dim:  Intermediate dimensionality in the prediction network."""

    def __init__(self, pool_size=8, use_post_corr=True):
        super().__init__()
        num_corr_channel = pool_size*pool_size
        self.use_post_corr = use_post_corr
        if use_post_corr:
            self.post_corr = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(1, 1), padding=0, stride=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=(1, 1), padding=0, stride=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 64, kernel_size=(1, 1), padding=0, stride=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
        self.channel_attention = SEModule(num_corr_channel,reduction=4)
        self.spatial_attention = nn.Sequential()

    def get_ref_kernel(self, feat1, bb1):
        assert bb1.dim() == 3

        # Extract first train sample
        assert len(feat1) == 1  # only support single feature level for now
        feat1 = feat1[0]

        sr = 2  # TODO: adaptive crop region according to search region rate
        self.ref_kernel = feat1[:, :,
                          feat1.shape[2]*(sr-1)//(sr*2):feat1.shape[2]*(sr+1)//(sr*2),
                          feat1.shape[3]*(sr-1)//(sr*2):feat1.shape[3]*(sr+1)//(sr*2)]

    def fuse_feat(self, feat2, convert_trt=False):
        """ fuse features from reference and test branch """
        assert len(feat2) == 1
        feat2 = feat2[0]

        # Step1: pixel-wise correlation
        #self.ref_kernel = torch.randn(1,256,8,8)
        #self.ref_kernel = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)
        feat_corr, _ = self.corr_fun(self.ref_kernel, feat2)

        # Step2: channel attention: Squeeze and Excitation
        if self.use_post_corr:
            # feat_corr = self.post_corr(feat_corr)
            if not convert_trt:
                feat_corr = self.post_corr(feat_corr)
            else:
                # tensorrt transform part 2
                post_corr_trt = torch2trt(self.post_corr, [feat_corr])
                feat_corr = post_corr_trt(feat_corr)
                torch.save(post_corr_trt.state_dict(), 'trt_models/trt_model_RF_RF_r34/post_corr_trt.pth')
                print('**********************TensorRT model 2 convert complete**********************')
        # feat_ca = self.channel_attention(feat_corr)
        if not convert_trt:
            feat_ca = self.channel_attention(feat_corr) # ???????????????????????????
        else:
            # tensorrt transform part 3
            channel_attention_trt = torch2trt(self.channel_attention, [feat_corr])
            feat_ca = channel_attention_trt(feat_corr)
            torch.save(channel_attention_trt.state_dict(), 'trt_models/trt_model_RF_RF_r34/channel_attention_trt.pth')
            print('**********************TensorRT model 3 convert complete**********************')
        return feat_ca

    def corr_fun(self, ker, feat):
        return self.corr_fun_mat(ker, feat)

    def corr_fun_mat(self, ker, feat):
        b, c, h, w = feat.shape
        ker = ker.reshape(b, c, -1).transpose(1, 2)
        feat = feat.reshape(b, c, -1)
        corr = torch.matmul(ker, feat)
        corr = corr.reshape(*corr.shape[:2], h, w)
        return corr, ker

    def corr_fun_loop(self, Kernel_tmp, Feature, KERs=None):
        b, c, _, _ = Kernel_tmp.shape
        size = Kernel_tmp.size()
        CORR = []
        Kernel = []
        for i in range(len(Feature)):
            ker = Kernel_tmp[i:i + 1]
            fea = Feature[i:i + 1]
            ker = ker.reshape(size[1], size[2] * size[3]).transpose(0, 1)
            ker = ker.unsqueeze(2).unsqueeze(3)
            if not (type(KERs) == type(None)):
                ker = torch.cat([ker, KERs[i]], 0)
            co = F.conv2d(fea, ker)
            CORR.append(co)
            ker = ker.unsqueeze(0)
            Kernel.append(ker)
        corr = torch.cat(CORR, 0)
        Kernel = torch.cat(Kernel, 0)
        return corr, Kernel

    def fuse_feat_trt(self, feat2):
        """ fuse features from reference and test branch """
        assert len(feat2) == 1
        feat2 = feat2[0]

        # Step1: pixel-wise correlation
        #self.ref_kernel = torch.randn(1,256,8,8)
        #self.ref_kernel = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)
        feat_corr, _ = self.corr_fun(self.ref_kernel, feat2)
        return feat_corr

#net =  PtCorr()
#y = torch.randn(1,256,16,16)
#model = net.fuse_feat(feat2= Variable(torch.unsqueeze(y, dim=0).float(), requires_grad=False))
#x = torch.randn(1,256,8,8)
#tensor_x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)
#y = torch.randn(1,256,32,32)
#tensor_y = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)
#flops, params = profile(net, inputs = (x,))
#flops, params = clever_format([flops, params], "%.3f")
#print(flops,'##',params)
