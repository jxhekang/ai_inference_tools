import torch
import numpy as np
from termcolor import cprint
from models import pytorch_models

class PytorchInfer:
    def __init__(self, args):
        cprint('PytorchInfer init', 'green')
        self.device = args.device
        self.model = pytorch_models[args.arch](pretrained=False, progress=False).to(self.device).eval()
            # print(self.resnet50_gpu)
        self.model.load_state_dict(torch.load(args.weights_dir), strict=True)
        # self.input_shape = xx
        # self.output_shape = xx

    def forward(self, input):
        with torch.no_grad():
            inputs = torch.from_numpy(input).to(self.device)
            outputs = np.array(self.model(inputs).cpu())
        return outputs
