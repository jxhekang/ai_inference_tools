import cv2
from termcolor import cprint

class OpenCvDnn:
    def __init__(self, args):
        self.net = cv2.dnn.readNetFromCaffe(args.net_dir, args.weights_dir)
        
    def __get_outputs_names(self, net):
        layer_names = net.getLayerNames()
        return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    def __call__(self, img):
        self.net.setInput(img)
        outs = self.net.forward(self.__get_outputs_names(self.net))
        return outs 

class OpencvDnnInfer:
    def __init__(self, args):
        cprint('OpencvDnnInfer init', 'green')
        cv2.setNumThreads(self.thread_num)
        self.opencvdnn = OpenCvDnn(args.net_dir, args.dir_weight, 224, 224)
        # self.input_shape = xx
        # self.output_shape = xx
        
    def forward(self, input):
        outputs = self.opencvdnn(input)
        return outputs
