import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
# from layers.functions.prior_box import PriorBox

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


    def init_gates(self):
        self.gates = nn.Parameter(
            torch.ones(self.conv.out_channels)
        )

    def get_gates(self):
        return [self.gates]

    def gated_forward(self, *input, **kwargs):
        out = self.conv(input[1])
        out = self.bn(out)
        out = self.gates.view(1, -1, 1, 1) * out
        return F.relu(out, inplace=True)
        

class EagleEyeFace3(nn.Module):

  def __init__(self, phase, num_classes, share_location=False, use_landmark=False):
    super(EagleEyeFace3, self).__init__()
    self.share_location = share_location
    self.phase = phase
    self.num_classes = num_classes
    self.use_landmark = use_landmark
    # self.pool = nn.AvgPool2d(kernel_size=3, stride=1,padding=1, ceil_mode=True)
    # self.conv0 = BasicConv2d(3, 1, kernel_size=3, stride=1, padding=1)

    self.conv1 = BasicConv2d(3, 4, kernel_size=3, stride=2, padding=1)

    self.conv2_dw = BasicConv2d(4, 4, kernel_size=3, stride=2, padding=1, groups=4)
    self.conv2_pw = BasicConv2d(4, 16, kernel_size=1, stride=1, padding=0)

    self.conv3_dw = BasicConv2d(16, 16, kernel_size=3, stride=2, padding=1, groups=16)
    self.conv3_pw = BasicConv2d(16, 32, kernel_size=1, stride=1, padding=0)

    self.conv4_dw = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
    self.conv4_pw = BasicConv2d(32, 32, kernel_size=1, stride=1, padding=0)

    self.conv5_dw = BasicConv2d(32, 32, kernel_size=3, stride=2, padding=1)
    self.conv5_pw = BasicConv2d(32, 64, kernel_size=1, stride=1, padding=0)

    self.conv6_1_dw = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.conv6_1_pw = BasicConv2d(64, 64, kernel_size=1, stride=1, padding=0)

    self.conv6_2_dw = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.conv6_2_pw = BasicConv2d(64, 64, kernel_size=1, stride=1, padding=0)

    self.conv6_3_dw = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.conv6_3_pw = BasicConv2d(64, 64, kernel_size=1, stride=1, padding=0)

    self.conv6_4_dw = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.conv6_4_pw = BasicConv2d(64, 64, kernel_size=1, stride=1, padding=0)

    self.conv7_dw = BasicConv2d(64, 64, kernel_size=3, stride=2, padding=1)
    self.conv7_pw = BasicConv2d(64, 64, kernel_size=1, stride=1, padding=0)

    self.conv8_dw = BasicConv2d(64, 64, kernel_size=3, stride=2, padding=1)
    self.conv8_pw = BasicConv2d(64, 128, kernel_size=1, stride=1, padding=0)

    self.conv9_dw = BasicConv2d(128, 128, kernel_size=3, stride=1, padding=1)
    self.conv9_pw = BasicConv2d(128, 128, kernel_size=1, stride=1, padding=0)

    self.conv10_pw_1 = BasicConv2d(128, 96, kernel_size=1, stride=1, padding=0)
    self.conv10_dw = BasicConv2d(96, 96, kernel_size=3, stride=2, padding=1)
    self.conv10_pw_2 = BasicConv2d(96, 192, kernel_size=1, stride=1, padding=0)

    if self.use_landmark:
         self.loc, self.conf, self.landm = self.multibox(self.num_classes)
    else:
        self.loc, self.conf = self.multibox(self.num_classes)

    if self.phase == 'test':
        if self.share_location:
            self.classify_action_fun = nn.Softmax(dim=-1)
        else:
            self.classify_action_fun  = nn.Sigmoid()

    if self.phase == 'train':
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

  def multibox(self, num_classes):
    loc_layers = []
    conf_layers = []
    landm_layers = []
    conf_layers += [nn.Conv2d(64, 21 * num_classes, kernel_size=3, padding=1, dilation=1)]
    conf_layers += [nn.Conv2d(128, 1 * num_classes, kernel_size=3, padding=1, dilation=1)]
    conf_layers += [nn.Conv2d(192, 1 * num_classes, kernel_size=3, padding=1, dilation=1)]
    if self.share_location:
        loc_layers += [nn.Conv2d(64, 21 * 4, kernel_size=3, padding=1, dilation=1)]
        loc_layers += [nn.Conv2d(128, 1 * 4, kernel_size=3, padding=1, dilation=1)]
        loc_layers += [nn.Conv2d(192, 1 * 4, kernel_size=3, padding=1, dilation=1)]
    else:
        loc_layers += [nn.Conv2d(64, 21 * 4 * num_classes, kernel_size=3, padding=1, dilation=1)]
        loc_layers += [nn.Conv2d(128, 1 * 4 * num_classes, kernel_size=3, padding=1, dilation=1)]
        loc_layers += [nn.Conv2d(192, 1 * 4 * num_classes, kernel_size=3, padding=1, dilation=1)]
    if self.use_landmark:
        landm_layers += [nn.Conv2d(64, 21 * 236 * num_classes, kernel_size=3, padding=1, dilation=1)]
        landm_layers += [nn.Conv2d(128, 1 * 236 * num_classes, kernel_size=3, padding=1, dilation=1)]
        landm_layers += [nn.Conv2d(192, 1 * 236 * num_classes, kernel_size=3, padding=1, dilation=1)]
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers), nn.Sequential(*landm_layers)
    else:
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)


  def forward(self, x):

    detection_sources = list()
    loc = list()
    conf = list()
    landm = list()
    # x = self.pool(x)
    # x = self.conv0(x)
    x = self.conv1(x)
    x = self.conv2_dw(x)
    x = self.conv2_pw(x)
    x = self.conv3_dw(x)
    x = self.conv3_pw(x)
    x = self.conv4_dw(x)
    x = self.conv4_pw(x)
    x = self.conv5_dw(x)
    x = self.conv5_pw(x)
    x = self.conv6_1_dw(x)
    x = self.conv6_1_pw(x)
    x = self.conv6_2_dw(x)
    x = self.conv6_2_pw(x)
    x = self.conv6_3_dw(x)
    x = self.conv6_3_pw(x)
    x = self.conv6_4_dw(x)
    x = self.conv6_4_pw(x)
    x = self.conv7_dw(x)
    x = self.conv7_pw(x)
    detection_sources.append(x)
    x = self.conv8_dw(x)
    x = self.conv8_pw(x)
    x = self.conv9_dw(x)
    x = self.conv9_pw(x)
    detection_sources.append(x)
    x = self.conv10_pw_1(x)
    x = self.conv10_dw(x)
    x = self.conv10_pw_2(x)
    detection_sources.append(x)
    if self.use_landmark:
        for (x, l, c, lm) in zip(detection_sources, self.loc, self.conf, self.landm):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            landm.append(lm(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        landm = torch.cat([o.view(o.size(0), -1) for o in landm], 1)

        loc_factor = 1 if self.share_location else self.num_classes
        if self.phase == "test":
            output = (loc.view(loc.size(0), -1, 4*loc_factor),
                    self.classify_action_fun(conf.view(conf.size(0), -1, self.num_classes)),
                    landm.view(landm.size(0), -1, 236*loc_factor))
            return output
        else:
            output = (loc.view(loc.size(0), -1, 4*loc_factor),
                    conf.view(conf.size(0), -1, self.num_classes),
                    landm.view(landm.size(0), -1, 236*loc_factor))
            return output
    else:
        for (x, l, c) in zip(detection_sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        loc_factor = 1 if self.share_location else self.num_classes
        if self.phase == "test":
            output = (loc.view(loc.size(0), -1, 4*loc_factor),
                    self.classify_action_fun(conf.view(conf.size(0), -1, self.num_classes)))
            return output
        else:
            output = (loc.view(loc.size(0), -1, 4*loc_factor),
                    conf.view(conf.size(0), -1, self.num_classes))
            return output

if __name__ == '__main__':
    torch.set_num_threads(1)
    model = EagleEyeFace3('train', 3).eval()
    img = torch.randn(1, 3, 640, 480)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # with torch.autograd.profiler.profile(enabled=True) as prof:
    #     for i in range(10):
    #         model(img)
    #     # summary(model, (3, 512, 512))
    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    from torchstat import stat
    # stat(model, (3, 320, 240))
    stat(model, (3, 640, 480))