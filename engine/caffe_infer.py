import caffe
import numpy as np
from termcolor import cprint

class CaffeInfer:
    def __init__(self, args):
        cprint('CaffeInfer init', 'green')
        self.net = caffe.Net(args.net_dir, args.weight_dir, caffe.TEST)
        # self.input_shape = xx
        # self.output_shape = xx

    def forward(self, input):
        self.net.blobs['data'].data[...] = input
        out = self.net.forward()
        outputs = out.get('fc_norm')
        return outputs