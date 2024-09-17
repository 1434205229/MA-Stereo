from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from .submodule import *
import math
import gc
import time
import timm


# Spatial Attention
class SA_Module(nn.Module):
    def __init__(self, input_nc, output_nc=1, ndf=16):
        super(SA_Module, self).__init__()
        self.attention_value = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            nn.Conv2d(ndf, output_nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_value = self.attention_value(x)
        return attention_value

class StereoDRNetRefinement(nn.Module):
    def  __init__(self):
        super(StereoDRNetRefinement, self).__init__()

        # Left and warped error
        in_channels = 6
        self.attention = SA_Module(input_nc=32)
        self.conv1 = conv2d(in_channels, 16)
        self.conv2 = conv2d(1, 16)  # on low disparity
        self.conv3 = conv2d(16, 16)
        self.conv_start = BasicConv(48, 48, kernel_size=3, padding=2, dilation=2)

        self.conv1a = BasicConv(48, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, dilation=2, padding=2)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, dilation=2, padding=2)

        self.deconv4a = Conv2x_1(128, 96, deconv=True)
        self.deconv3a = Conv2x_1(96, 64, deconv=True)
        self.deconv2a = Conv2x_1(64, 48, deconv=True)
        self.deconv1a = Conv2x_1(48, 48, deconv=True)

        self.conv1b = Conv2x_1(48, 48)
        self.conv2b = Conv2x_1(48, 64)
        self.conv3b = Conv2x_1(64, 96, mdconv=True)
        self.conv4b = Conv2x_1(96, 128, mdconv=True)

        self.deconv4b = Conv2x_1(128, 96, deconv=True)
        self.deconv3b = Conv2x_1(96, 64, deconv=True)
        self.deconv2b = Conv2x_1(64, 48, deconv=True)
        self.deconv1b = Conv2x_1(48,48, deconv=True)

        self.final_conv = nn.Conv2d(48, 1, 3, 1, 1)
        # self.dilation_list = [1, 2, 4, 8, 1, 1]
        # self.dilated_blocks = nn.ModuleList()
        #
        # for dilation in self.dilation_list:
        #     self.dilated_blocks.append(BasicBlock(32, 32, stride=1, dilation=dilation))

        # self.dilated_blocks = nn.Sequential(*self.dilated_blocks)
        #
        # self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, left_img, right_img, left_disp,spx_pred1):

        # Warp right image to left view with current disparity
        # left_disp = F.interpolate(left_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
        left_disp=left_disp.unsqueeze(1)
        left_disp = left_disp * 4

        recon_left_img = reconstruction(right_img, left_disp.squeeze(1))[0]  # [B, C, H, W]
        #recon_left_img = disp_warp(right_img, left_disp)[0]

        error = recon_left_img - left_img  # [B, C, H, W]

        concat1 = torch.cat((error, left_img), dim=1)  # [B, 6, H, W]

        conv1 = self.conv1(concat1)  # [B, 16, H, W]
        conv2 = self.conv2(left_disp)  # [B, 16, H, W]
        conv3 = self.conv3(spx_pred1)
        concat2 = torch.cat((conv1, conv2), dim=1)  # [B, 32, H, W]

        attention_map = self.attention(concat2)
        concat2 = attention_map * torch.cat((concat2,conv3), dim=1)  # [B, 48, H, W]

        # out = self.dilated_blocks(concat2)  # [B, 32, H, W]
        # residual_disp = self.final_conv(out)  # [B, 1, H, W]
        x = self.conv_start(concat2)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)

        x = self.deconv4b(x, rem3)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)  # [B, 32, H, W]

        residual_disp = self.final_conv(x)  # [B, 1, H, W]

        disp = F.leaky_relu(left_disp + residual_disp, inplace=True)  # [B, 1, H, W](一般为relu)
        
        return disp

class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Feature(SubModule):
    def __init__(self):
        super(Feature, self).__init__()
        pretrained =  True
        model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True)
        layers = [1,2,3,5,6]
        chans = [16, 24, 32, 96, 160]
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1

        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv_stem(x)))  #(1,32,H/2,W/2)
        x2 = self.block0(x)#(1,16,H/2,W/2)
        x4 = self.block1(x2)#(1,24,H/4,W/4)
        # return x4,x4,x4,x4
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)
        return [x4, x8, x16, x32]

class FeatUp(SubModule):
    def __init__(self):
        super(FeatUp, self).__init__()
        chans = [16, 24, 32, 96, 160]
        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x(chans[3]*2, chans[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x(chans[2]*2, chans[1], deconv=True, concat=True)
        self.conv4 = BasicConv(chans[1]*2, chans[1]*2, kernel_size=3, stride=1, padding=1)

        self.weight_init()

    def forward(self, featL, featR=None):
        x4, x8, x16, x32 = featL

        y4, y8, y16, y32 = featR
        x16 = self.deconv32_16(x32, x16)
        y16 = self.deconv32_16(y32, y16)
        
        x8 = self.deconv16_8(x16, x8)
        y8 = self.deconv16_8(y16, y8)

        x4 = self.deconv8_4(x8, x4)
        y4 = self.deconv8_4(y8, y4)
        
        x4 = self.conv4(x4)
        y4 = self.conv4(y4)

        return [x4, x8, x16, x32], [y4, y8, y16, y32]


class Context_Geometry_Fusion(SubModule):
    def __init__(self, cv_chan, im_chan):
        super(Context_Geometry_Fusion, self).__init__()

        self.semantic = nn.Sequential(
            BasicConv(im_chan, im_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(im_chan//2, cv_chan, 1))
        self.att1 = nn.Sequential(BasicConv(cv_chan, cv_chan, is_3d=True, bn=True, relu=True, kernel_size=(1, 3, 3),
                                           padding=(0, 1, 1), stride=1, dilation=1),
                                 nn.Conv3d(cv_chan, cv_chan, kernel_size=1, stride=1, padding=0, bias=False))
        self.att2 = nn.Sequential(BasicConv(cv_chan, cv_chan, is_3d=True, bn=True, relu=True, kernel_size=(1,5,5),
                                 padding=(0,2,2), stride=1, dilation=1),
                                 nn.Conv3d(cv_chan, cv_chan, kernel_size=1, stride=1, padding=0, bias=False))
        self.att3 = nn.Sequential(BasicConv(cv_chan, cv_chan, is_3d=True, bn=True, relu=True, kernel_size=(1, 7, 7),
                                           padding=(0, 3, 3), stride=1, dilation=1),
                                 nn.Conv3d(cv_chan, cv_chan, kernel_size=1, stride=1, padding=0, bias=False))
        self.agg = BasicConv(cv_chan, cv_chan, is_3d=True, bn=True, relu=True, kernel_size=(1,5,5),
                             padding=(0,2,2), stride=1, dilation=1)
        self.last_conv = nn.Sequential(
            BasicConv(3*cv_chan, cv_chan, is_3d=True, bn=True, relu=True, kernel_size=(1, 1, 1),
                      padding=(0, 0, 0), stride=1, dilation=1),
            nn.Conv3d(cv_chan, cv_chan, kernel_size=1, stride=1, padding=0, bias=False))

        self.se = SEWeightModule(im_chan)
        self.weight_init()

    def forward(self, cv, feat):
        '''
        '''

        # feat = self.semantic(feat).unsqueeze(2)  #（1，48，1，8，16）
        feat = self.se(feat) * feat
        feat = self.semantic(feat)
        feat = feat.unsqueeze(2)  # （1，48，1，8，16）

        att1 = self.att1(feat+cv)
        b, c, d, h, w = att1.size()
        att2 = self.att2(feat+cv)
        att3 = self.att3(feat+cv)
        att=torch.stack((att1,att2,att3),dim=1)
        att=att.reshape(b,3*c,d,h,w)
        att = self.last_conv(att)
        cv = torch.sigmoid(att)*feat + cv
        cv = self.agg(cv)
        return cv



class hourglass_fusion(nn.Module):
    def __init__(self, in_channels):
        super(hourglass_fusion, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 1, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))


        self.CGF_32 = Context_Geometry_Fusion(in_channels*6, 160)
        self.CGF_16 = Context_Geometry_Fusion(in_channels*4, 192)
        self.CGF_8 = Context_Geometry_Fusion(in_channels*2, 64)

    def forward(self, x, imgs):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        conv3 = self.CGF_32(conv3, imgs[3])
        conv3_up = self.conv3_up(conv3)

        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)

        conv2 = self.CGF_16(conv2, imgs[2])
        conv2_up = self.conv2_up(conv2)

        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)

        conv1 = self.CGF_8(conv1, imgs[1])
        conv = self.conv1_up(conv1)

        return conv


class CGI_Stereo(nn.Module):
    def __init__(self, maxdisp):
        super(CGI_Stereo, self).__init__()
        self.maxdisp = maxdisp 
        self.feature = Feature()
        self.feature_up = FeatUp()
        chans = [16, 24, 32, 96, 160]
        self.sobelx = MySobelx()
        self.stem_2 = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx1 = nn.Sequential(nn.ConvTranspose2d(2 * 32, 16, kernel_size=4, stride=2, padding=1), )
        self.spx_2 = Conv2x(32, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(97, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
            )

        self.conv = BasicConv(97, 48, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(48, 48, kernel_size=1, padding=0, stride=1)
        self.semantic = nn.Sequential(
            BasicConv(97, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 16, kernel_size=1, padding=0, stride=1, bias=False))
        self.agg = BasicConv(16, 16, is_3d=True, kernel_size=(1,5,5), padding=(0,2,2), stride=1)
        self.hourglass_fusion = hourglass_fusion(16)
        self.corr_stem = BasicConv(1, 16, is_3d=True, kernel_size=3, stride=1, padding=1)

        self.RefineModule = StereoDRNetRefinement()
    def get_edge(self, input):
            output = self.sobelx(input)
            return output
    def forward(self, left, right):
        features_left = self.feature(left)
        features_right = self.feature(right)
        features_left, features_right = self.feature_up(features_left, features_right)
        stem_2x = self.stem_2(left)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(right)
        stem_4y = self.stem_4(stem_2y)

        B, C, H, W = stem_4x.size()
        left_edge = self.get_edge(left)  # torch.Size([B, 1, 256, 512])
        right_edge = self.get_edge(right)
        left_edge = F.interpolate(left_edge, size=(H, W), mode='bilinear')
        right_edge = F.interpolate(right_edge, size=(H, W), mode='bilinear')

        features_left[0] = torch.cat((features_left[0], stem_4x, left_edge), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y, right_edge), 1)


        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))

        corr_volume = build_norm_correlation_volume(match_left, match_right, self.maxdisp//4)
        corr_volume = self.corr_stem(corr_volume)
        feat_volume = self.semantic(features_left[0]).unsqueeze(2)
        volume = self.agg(feat_volume * corr_volume)  #(B,16,D/4,H/4,W/4)
        cost = self.hourglass_fusion(volume, features_left)  #(B,1,D/4,H/4,W/4)

        xspx = self.spx_4(features_left[0]) #(1,32,64,128)
        xspx = self.spx_2(xspx, stem_2x)#(1,64,128,256)
        spx_pred = self.spx(xspx) #(1,9,256,512)
        spx_pred1 = self.spx1(xspx)  # (1,16,256,512)
        spx_pred = F.softmax(spx_pred, 1) #(1,9,256,512)

        disp_samples = torch.arange(0, self.maxdisp//4, dtype=cost.dtype, device=cost.device) #48
        disp_samples = disp_samples.view(1, self.maxdisp//4, 1, 1).repeat(cost.shape[0],1,cost.shape[3],cost.shape[4])#(1,48,64,128)
        pred = regression_topk(cost.squeeze(1), disp_samples, 2)#(1,1,64,128)
        # #超像素”权重[34]来向上采样视差图d_0到原始分辨率
        pred_up= context_upsample(pred, spx_pred)
        #(1,256,512)
        refine_disp = self.RefineModule(left, right, pred_up,spx_pred1)

        if self.training:
            return [refine_disp.squeeze(1), pred.squeeze(1)*4]

        else:
            return [refine_disp.squeeze(1)]
