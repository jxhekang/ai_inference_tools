from termcolor import cprint
import argparse

from data import DataLoader
from engine import *
from models import *

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_dir',
                        type=str,
                        help='net file')
    parser.add_argument('--weights_dir',
                        type=str,
                        help='weights file')
    parser.add_argument('--runtime_type',
                        type=str,
                        help='select your inference tool')
    parser.add_argument('--device',
                        default='cpu',
                        type=str,
                        help='select your hardware device')
    parser.add_argument('--arch',
                        default='resnet18',
                        type=str,
                        help='only valid for pytorch')
    parser.add_argument('--input_synthetic',
                        default='True',
                        type=str,
                        help='synthetic or image')
    parser.add_argument('--input_image_dir',
                        default='data/123.jpg',
                        type=str,
                        help='input_image_dir')
    args = parser.parse_args()
    return args

class AiInferenceTools:
    def __init__(self, args):
        cprint('=> loading net_dir : {}'.format(args.net_dir), 'green')
        cprint('=> loading weight_dir : {}'.format(args.weights_dir), 'green')

        if(args.runtime_type == "pytorch"):
            self.engine = PytorchInfer()
        elif(args.runtime_type == "caffe"):
            self.engine = CaffeInfer()
        elif(args.runtime_type == "opencvdnn"):
            self.engine = OpencvDnnInfer()
        elif(args.runtime_type == "onnxruntime"):
            self.engine = OnnxRuntimeInfer()
        elif(args.runtime_type == "openvino"):
            self.engine = OpenVinoInfer()
        elif(args.runtime_type == "tensorrt"):
            self.engine = TensorRtInfer()
        else:
            cprint("ai_inference_tools don't support : {}".format(args.runtime_type), 'red')
        
        self.input_shape = self.engine.input_shape
        
    def forward(self):
        return self.engine.forward()



if __name__ == "__main__":
    args = parse_args()

    ai_inference_tools = AiInferenceTools(args)

    input = DataLoader(args, ai_inference_tools.input_shape)

    res = ai_inference_tools.forward(input)

    print(res)