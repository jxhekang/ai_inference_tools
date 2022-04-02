from termcolor import cprint
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import json
import time
from data.dataloader import DataLoader
from engine import *
from models import *
from tools.tools import load_labels, preprocess, postprocess

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
            self.engine = OnnxRuntimeInfer(args.net_dir, 1)
        elif(args.runtime_type == "openvino"):
            self.engine = OpenVinoInfer()
        elif(args.runtime_type == "tensorrt"):
            self.engine = TensorRtInfer()
        else:
            cprint("ai_inference_tools don't support : {}".format(args.runtime_type), 'red')
        
        # self.input_shape = self.engine.input_shape
        
    def forward(self, input):
        return self.engine.forward(input)



if __name__ == "__main__":
    args = parse_args()

    ai_inference_tools = AiInferenceTools(args)
    
    # input_shape = [1, 3, 224, 224]

    # my_dataloder = DataLoader(args, input_shape)
    
    # input = my_dataloder()

    labels = load_labels('data/imagenet-simple-labels.json')
    image = Image.open('data/cat1.jpg')
    # image = Image.open('data/dog1.jpg')
    # image = Image.open('data/eagle.jpg')
    image = image.resize((224, 224))
    
    # image = cv2.imread('data/dog1.jpg')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (224, 224))
    
    img_np = np.array(image, dtype=np.float32)
    image_data = np.array(img_np).transpose(2, 0, 1)
    input_data = preprocess(image_data)

    start = time.time()
    raw_result = ai_inference_tools.forward(input_data)
    end = time.time()
    
    res = postprocess(raw_result)

    inference_time = np.round((end - start) * 1000, 2)
    idx = np.argmax(res)

    print('========================================')
    print('Final top prediction is: ' + str(res[idx]))
    print('========================================')

    print('========================================')
    print('Final top prediction is: ' + labels[idx])
    print('========================================')

    print('========================================')
    print('Inference time: ' + str(inference_time) + " ms")
    print('========================================')

    sort_idx = np.flip(np.squeeze(np.argsort(res)))
    print('============ Top 5 labels are: ============================')
    print(labels[sort_idx[:5]])
    print('===========================================================')