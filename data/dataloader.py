import numpy as np
import cv2
from termcolor import cprint


class DataLoader:
    def __init__(self, args, input_shape):
        self.input_synthetic = args.input_synthetic
        self.filename = args.input_image_dir
        self.input_shape = input_shape #nchw

        
    def get_img_np_nchw(filename, height, width):
        image = cv2.imread(filename)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height))
        # miu = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        miu = np.array([0, 0, 0])
        std = np.array([1, 1, 1])
        img_np = np.array(image, dtype=float) / 255.
        # img_np = np.array(image, dtype=float)
        r = (img_np[:, :, 0] - miu[0]) / std[0]
        g = (img_np[:, :, 1] - miu[1]) / std[1]
        b = (img_np[:, :, 2] - miu[2]) / std[2]
        img_np_t = np.array([r, g, b])
        img_np_nchw = np.expand_dims(img_np_t, axis=0)
        print(img_np_nchw.shape)
        return img_np_nchw

    def __call__(self):
        if(self.input_synthetic):
            output = np.ones(self.input_shape)
        else:# image
            output = self.get_img_np_nchw(self.filename, self.input_shape[2], self.input_shape[3])
        return output
        

