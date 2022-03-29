import torch
import torch.nn as nn
import onnx
import torchvision.models as models
import self_define_models
import numpy as np

from termcolor import cprint
import argparse

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',
                        default='resnet18',
                        type=str,
                        help='only valid for pytorch')
    parser.add_argument('--pretrained',
                        default=False,
                        type=bool,
                        help='only valid for pytorch')
    parser.add_argument('--self_define',
                        default=False,
                        type=bool,
                        help='only valid for pytorch')
    parser.add_argument('--ckpt',
                        default='./ckpt/ckpt.pth',
                        type=str,
                        help='only valid for pytorch')
    parser.add_argument('--model_type',
                        default='onnx',
                        type=str,
                        help='torchscript or onnx')
    args = parser.parse_args()
    return args

mods = []

# def leaf_modules(model):
#     try:
#         for c in model.children():
#             # print(c)
#             leaf_modules(c)
#     except Exception as ret:
#         print("111222333")
#         for m in model.modules():
#             mods.append(m)
#             print(m)

def leaf_modules(model):
    for c in model.children():
        # print(c)
        leaf_modules(c)

def accumulate_params(self):
    if is_supported_instance(self):
        return self.__mems__
    else:
        sum = 0
        for m in self.children():
            sum += m.accumulate_params()
        return sum

# 模型显存占用监测函数
# model：输入的模型
# input：实际中需要输入的Tensor变量
# type_size 默认为 4 默认类型为 float32 

def modelsize(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    # print("model == ", model)
    # mods = list(model.modules())
    leaf_modules(model)

    print("mods == ", mods)
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        # print("m == ", m)
        # print("input_.size() == ", input_.size())
        # out = m(input_)
        # print("out.size() == ", out.size())
        # out_sizes.append(np.array(out.size()))
        # input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums


    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))

if __name__ == "__main__":
    args = parse_args()
    
    if args.self_define:
        # inputs = torch.ones((1,256,1,1))
        # model = self_define_models.__dict__[args.arch](256, 4, 256, 2)
        inputs = torch.ones((1,3,640,640))
        # model = self_define_models.__dict__[args.arch](phase='test', num_classes=3)
        model = torch.jit.load("./self_define_models/" + args.arch + ".pt")
    else:
        # model = torch.hub.load('pytorch/vision', args.arch, pretrained=args.pretrained)
        inputs = torch.ones((1,3,224,224))
        if args.arch == 'resnet18':
            model = models.resnet18()
        elif args.arch == 'resnet50':
            model = models.resnet50()
        elif args.arch == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=False)
        else:
            pass
    print('11111')
    model.eval()

    # modelsize(model, inputs)
    # from torchstat import stat
    # stat(model, (3, 640, 640))

    if args.pretrained:
        model.load_state_dict(args.ckpt)
    if args.model_type == 'onnx':
        onnx_name = args.arch + ".onnx"
        # torch.onnx.export(model,                       # model being run
        #                   inputs,                         # model input (or a tuple for multiple inputs)
        #                   onnx_name,           # where to save the model (can be a file or file-like object)
        #                   export_params=True,        # store the trained parameter weights inside the model file
        #                   opset_version=12,          # the ONNX version to export the model to
        #                   do_constant_folding=True  # whether to execute constant folding for optimization
        #                 #   input_names = ['input'],   # the model's input names
        #                 #   output_names = ['1250', '1295', '1294'],
        #                 #   dynamic_axes={'input': {0: 'n', 1: 'c', 2: 'h', 3: 'w'}, # support dynamic reshape
        #                 #                 'output': {0: 'n', 1: 'c', 2: 'h', 3: 'w'}}, # support dynamic reshape
        #                 # example_outputs=out
        #                 ) # the model's output names
        torch.onnx.export(model,                       # model being run
                          inputs,                         # model input (or a tuple for multiple inputs)
                          onnx_name,           # where to save the model (can be a file or file-like object)
                          export_params=True,        # store the trained parameter weights inside the model file
                          opset_version=12,          # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names = ['input'],   # the model's input names
                          dynamic_axes={'input': {0: 'n', 1: 'c', 2: 'h', 3: 'w'}} # support dynamic reshape
                        # example_outputs=out
                        ) # the model's output names
    else:
        print(inputs.size())
        result = model(inputs)
        # print(result)
        traced_script_module = torch.jit.trace(model, inputs)
        traced_script_module.save(args.arch + ".pt")
        
        # model_new = torch.jit.load(args.arch + ".pth")
        # result = model_new(inputs)
        # print(model_new)
        # traced_script_module = torch.jit.script(model, inputs)
        # traced_script_module.save(args.arch + ".pth")