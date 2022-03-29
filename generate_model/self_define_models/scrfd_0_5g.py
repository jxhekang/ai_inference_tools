import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
# from torchsummary import summary
import math
from ptflops import get_model_complexity_info

class SCRFD(nn.Module):

    def conv_bn(self, inp, oup, stride, activation=True):
        if activation:
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.LeakyReLU(inplace=True))
        else:
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup))

    def conv_dw(self, inp, oup, stride, activation=True):
        if activation:
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.LeakyReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.LeakyReLU(inplace=True))
        else:
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.LeakyReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup))

    def __init__(self, cfg = None, phase = 'train'):
        super(SCRFD, self).__init__()
        self.phase = phase
        #stage_planes = [16, 16, 40, 72, 152, 288]  #0.25 default
        #stage_blocks = [2, 3, 2, 6]
        self.stem = nn.Sequential(self.conv_bn(3, 16, 2), self.conv_dw(16, 16, 1))
        self.stage_layers = []

        #stage0
        self.layer0_0 = self.conv_dw(16, 40, 2)
        self.layer0_1 = self.conv_dw(40, 40, 1)

        #stage1
        self.layer1_0 = self.conv_dw(40, 72, 2)
        self.layer1_1 = self.conv_dw(72, 72, 1)
        self.layer1_2 = self.conv_dw(72, 72, 1)
        
        #stage2
        self.layer2_0 = self.conv_dw(72, 152, 2)
        self.layer2_1 = self.conv_dw(152, 152, 1)
        
        #stage3
        self.layer3_0 = self.conv_dw(152, 288, 2)
        self.layer3_1 = self.conv_dw(288, 288, 1)
        self.layer3_2 = self.conv_dw(288, 288, 1)
        self.layer3_3 = self.conv_dw(288, 288, 1)
        self.layer3_4 = self.conv_dw(288, 288, 1)
        self.layer3_5 = self.conv_dw(288, 288, 1)

        self.Conv_78 = self.conv_bn(72, 16, 1, False)
        self.Conv_79 = self.conv_bn(152, 16, 1, False)
        self.Conv_80 = self.conv_bn(288, 16, 1, False)

        self.Conv_121 = self.conv_bn(16, 16, 1, False)
        self.Conv_122 = self.conv_bn(16, 16, 1, False)
        self.Conv_123 = self.conv_bn(16, 16, 1, False)
        self.Conv_124 = self.conv_bn(16, 16, 2, False)
        self.Conv_126 = self.conv_bn(16, 16, 2, False)
        self.Conv_128 = self.conv_bn(16, 16, 1, False)
        self.Conv_129 = self.conv_bn(16, 16, 1, False)

        self.Conv_130 = self.conv_dw(16, 64, 1)
        self.Conv_132 = self.conv_dw(64, 64, 1)

        self.Conv_134 = self.conv_bn(64, 3*2, 1, False)
        self.Conv_135 = self.conv_bn(64, 3*4*2, 1, False)
        # self.Conv_137 = BasicConv2d(64, 20, 1)

        self.Conv_148 = self.conv_dw(16, 64, 1)
        self.Conv_150 = self.conv_dw(64, 64, 1)

        self.Conv_152 = self.conv_bn(64, 3*2, 1, False)
        self.Conv_153 = self.conv_bn(64, 3*4*2, 1, False)
        # self.Conv_155 = BasicConv2d(64, 20, 1)

        self.Conv_166 = self.conv_dw(16, 64, 1)
        self.Conv_168 = self.conv_dw(64, 64, 1)

        self.Conv_170 = self.conv_bn(64, 3*2, 1, False)
        self.Conv_171 = self.conv_bn(64, 3*4*2, 1, False)
        # self.Conv_173 = BasicConv2d(64, 20, 1)


    def forward(self, x):
        x = self.stem(x)

        x = self.layer0_0(x)
        x = self.layer0_1(x)

        x = self.layer1_0(x)
        x = self.layer1_1(x)
        x1 = self.layer1_2(x)

        x = self.layer2_0(x1)
        x2 = self.layer2_1(x)

        x = self.layer3_0(x2)
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.layer3_3(x)
        x = self.layer3_4(x)
        x3 = self.layer3_5(x)
       
        feat0 = self.Conv_78(x1)
        feat1 = self.Conv_79(x2)
        feat2 = self.Conv_80(x3)

        up2 = F.interpolate(feat2, size=[feat1.size(2), feat1.size(3)], mode="nearest")
        feat1 = feat1 + up2
        up1 = F.interpolate(feat1, size=[feat0.size(2), feat0.size(3)], mode="nearest")
        feat0 = feat0 + up1

        feat0 = self.Conv_121(feat0)

        feat0_0 = self.Conv_130(feat0)
        feat0_0 = self.Conv_132(feat0_0)

        feat0_conf = self.Conv_134(feat0_0).permute(0,2,3,1)
        feat0_loc = self.Conv_135(feat0_0).permute(0,2,3,1)
        # feat0_landm = self.Conv_137(feat0_0).permute(0,2,3,1)

        feat0_1 = self.Conv_124(feat0)
        feat1 = self.Conv_122(feat1)
        feat1 = feat1 + feat0_1

        feat1_0 = self.Conv_128(feat1)
        feat1_0 = self.Conv_148(feat1_0)
        feat1_0 = self.Conv_150(feat1_0)

        feat1_conf = self.Conv_152(feat1_0).permute(0,2,3,1)
        feat1_loc = self.Conv_153(feat1_0).permute(0,2,3,1)
        # feat1_landm = self.Conv_155(feat1_0).permute(0,2,3,1)

        feat1_1 = self.Conv_126(feat1)
        feat2 = self.Conv_123(feat2)
        feat2 = feat2 + feat1_1
        feat2_0 = self.Conv_129(feat2)
        feat2_0 = self.Conv_166(feat2_0)
        feat2_0 = self.Conv_168(feat2_0)

        feat2_conf = self.Conv_170(feat2_0).permute(0,2,3,1)
        feat2_loc = self.Conv_171(feat2_0).permute(0,2,3,1)
        # feat2_landm = self.Conv_173(feat2_0).permute(0,2,3,1)

        feat0_conf = torch.reshape(feat0_conf, (feat0_conf.shape[0], -1, 3))
        feat0_loc = torch.reshape(feat0_loc, (feat0_loc.shape[0], -1, 12))
        # feat0_landm = torch.reshape(feat0_landm, (feat0_landm.shape[0], -1, 10))

        feat1_conf = torch.reshape(feat1_conf, (feat1_conf.shape[0], -1, 3))
        feat1_loc = torch.reshape(feat1_loc, (feat1_loc.shape[0], -1, 12))
        # feat1_landm = torch.reshape(feat1_landm, (feat1_landm.shape[0], -1, 10))

        feat2_conf = torch.reshape(feat2_conf, (feat2_conf.shape[0], -1, 3))
        feat2_loc = torch.reshape(feat2_loc, (feat2_loc.shape[0], -1, 12))
        # feat2_landm = torch.reshape(feat2_landm, (feat2_landm.shape[0], -1, 10))

        bbox_regressions = torch.cat([feat0_loc, feat1_loc, feat2_loc], 1)
        classifications = torch.cat([feat0_conf, feat1_conf, feat2_conf], 1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications)
        else:
            # output = (bbox_regressions, F.softmax(classifications, dim=-1))
            output = (bbox_regressions, torch.sigmoid(classifications))
        return output
if __name__ == '__main__':
    torch.set_num_threads(1)
    model = SCRFD().eval()
    img = torch.randn(1, 3, 640, 480)
    # print(model.dtype)
    print(img.dtype)
    macs, params = get_model_complexity_info(model, (3, 440, 330), as_strings=True,
                                                print_per_layer_stat=True, verbose=True)
    # macs, params = get_model_complexity_info(model, (3, 640, 480), as_strings=True,
    #                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))    

    from torchstat import stat
    stat(model, (3, 440, 330))
    # stat(model, (3, 640, 480))
    # with torch.autograd.profiler.profile(enabled=True) as prof:
    #     for i in range(10):
    #         model(img)
    #     # summary(model, (3, 640, 480))
    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    # with torch.cuda.device(0):
        # macs, params = get_model_complexity_info(model, (3, 320, 240), as_strings=True,
        #                                         print_per_layer_stat=True, verbose=True)
        # macs, params = get_model_complexity_info(model, (3, 640, 480), as_strings=True,
        #                                         print_per_layer_stat=True, verbose=True)
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
