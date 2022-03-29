# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
# import torch.onnx
import os
import time
import sys
# import cv2
# import torchvision.models as models
# import torch
import onnx
import onnxruntime
# from openvino.inference_engine import IECore
# import cv2
# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit
import logging as log

# from skimage import io
# from skimage.transform import resize
# from matplotlib import pyplot as plt
import numpy as np
sys.path.append("/workspace/lib")
sys.path.append("/ssd/khe/git_src/caffe_all/RefineDet/build_x64_cuda/install/python")

# import caffe
########################################################################################
# set self-design model
# sys.path.append("..") 
# sys.path.insert(0, 'camera_checker_v2')
# import blur_model.models as models
# from blur_model.datasets import *
# from blur_model.utils import *
# from blur_model.paths import *
# from blur_model.models.resskspp_vino.resskspp_vino_infer import Resskspp_Vino
# os.environ["OMP_NUM_THREADS"] = "2"
########################################################################################
########################################################################################
# set runtime_type
# runtime_type = 0 # pytorch
# runtime_type = 1 # openvino
runtime_type = 2 # onnx runtime
# runtime_type = 3 # opencv dnn
# runtime_type = 4 # tensorrt
# runtime_type = 5 # caffe
########################################################################################
# set thread_num
thread_num = 8
# set BATCH_SIZE
BATCH_SIZE = 1
########################################################################################
# set model path

# model-zyw
model_onnx = 'v080700-drop0.onnx'
# model_onnx = 'v080700.onnx'
# model_trt = 'zhangyongwei-x86_64.trt7'


# model_trt = 'v080700-pt1.6.1-op11.trt7'
# model_trt = 'v080700-pt1.6.1-op11.trt8'
im_list =  np.ones([1, 3, 108, 108]).astype(np.float32)
# im_list =  np.zeros([1, 3, 108, 108]).astype(np.float32)

# resnet50
# weight_pth = 'resnet50-19c8e357.pth'
# model_onnx = 'resnet50_pytorch.onnx'
# # model_onnx = 'resnet50.onnx'
# model_trt = 'resnet50_pytorch.trt7'
# model_trt = 'resnet50_pytorch.trt8'
# im_list =  np.ones([1, 3, 224, 224]).astype(np.float32)

# yolov5_v0.4
# model_openvino = 'yolov5/yolov5l_v0.4.xml'
# model_onnx = 'yolov5/yolov5l_v0.4.onnx'
# im_list =  np.ones([1, 3, 640, 640]).astype(np.float32)

# yolov5_v0.5
# model_openvino = 'yolov5_new/yolov5l_v0.5.xml'
# model_onnx = 'yolov5_new_onnx/yolov5l_v0.5.onnx'
# im_list =  np.ones([1, 3, 640, 640]).astype(np.float32)

# hrNet
# model_openvino = 'hrNet/hrnetv2_w18_v0.3.xml'
# model_onnx = 'hrNet/hrnetv2_w18_v0.3.onnx'
# im_list =  np.ones([1, 3, 320, 320]).astype(np.float32)

# car_color
# model_openvino = 'resnet18_color/car_color_reid_resnet18_4s_shop_v0.11.xml'
# model_prototxt = 'resnet18_color/car_color_reid_resnet18_4s_shop_v0.11.prototxt'
# model_caffemodel = 'resnet18_color/car_color_reid_resnet18_4s_shop_v0.11.caffemodel'
# im_list =  np.ones([1, 3, 224, 224]).astype(np.float32)

# resnet18_car_type
# model_openvino = 'resnet18_car_type/car_model_reid_resnet18_4s_shop_v0.11.xml'
# model_prototxt = 'resnet18_car_type/car_model_reid_resnet18_4s_shop_v0.11.prototxt'
# model_caffemodel = 'resnet18_car_type/car_model_reid_resnet18_4s_shop_v0.11.caffemodel'
# im_list =  np.ones([1, 3, 224, 224]).astype(np.float32)
########################################################################################
# sub-function
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

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

########################################################################################
# sub-class
class OpenVinoInfer():
    def __init__(self, model_path, thread_num):
        xml_dir = os.path.dirname(os.path.abspath(__file__))
        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.ERROR, stream=sys.stdout)
        model_xml = model_path
        model_bin = model_path.strip('.xml') + ".bin"
        print("model_bin ==", model_bin)
        # Plugin initialization for specified device and load extensions library if specified
        log.info("Creating Inference Engine")
        ie = IECore()
        ie.set_config({"CPU_THREADS_NUM" : str(thread_num)}, "CPU")
        print(ie.get_config("CPU", "CPU_THREADS_NUM"))
        # Read IR
        log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
        net = ie.read_network(model=model_xml, weights=model_bin)
        #reshape
        input_layer = next(iter(net.inputs))
        # n, c, h, w = net.inputs[input_layer]
        # net.reshape({input_layer: (input_shapes[0], input_shapes[1], input_shapes[2], input_shapes[3])})

        # if "CPU" in "CPU":
            # supported_layers = ie.query_network(net, "CPU")
            # not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
            # if len(not_supported_layers) != 0:
            #     log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
            #               format("CPU", ', '.join(not_supported_layers)))
            #     log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
            #               "or --cpu_extension command line argument")
            #     sys.exit(1)
        assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
        # assert len(net.outputs) == 1, "Sample supports only single output topologies"
        self.input_blob = next(iter(net.inputs))
        self.output_blob = next(iter(net.outputs))
        # Loading model to the plugin
        log.info("Loading model to the plugin")
        self.exec_net = ie.load_network(network=net, device_name="CPU")

    def forward(self, x):
        # Start sync inference
        log.info("Starting inference in synchronous mode")
        res = self.exec_net.infer(inputs={self.input_blob: x})
        return res
# sub-class
class OpenCvDnn:
    def __init__(self, model, weights, input_width, input_height):
        self.net = cv2.dnn.readNetFromCaffe(model, weights)
        self.input_width = input_width
        self.input_height = input_height
        
    def __get_outputs_names(self, net):
        layer_names = net.getLayerNames()
        return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    def __norm_vec(self, vec):
        vec_l2_norm = np.linalg.norm(vec)
        vec = vec / vec_l2_norm
        return vec 
    
    def __call__(self, img):
        # blob = cv2.dnn.blobFromImage(img, 1/255., (self.input_height, self.input_width), (0, 0, 0), False)
        self.net.setInput(img)
        outs = self.net.forward(self.__get_outputs_names(self.net))
        return outs 

########################################################################################
# save model
# def save_model_torch2onnx():
#     model = torchvision.models.resnet50(pretrained=True).cpu()
#     x = torch.randn(1, 3, 224, 224)

#     for name, param in model.named_parameters():
#     	print(name, '      ', param.size())

#     out = model(x) # example_outputs must be provided when exporting a ScriptModule

#     torch.onnx.export(model,                       # model being run
#                       x,                         # model input (or a tuple for multiple inputs)
#                       "resnet50.onnx",           # where to save the model (can be a file or file-like object)
#                       export_params=True,        # store the trained parameter weights inside the model file
#                       opset_version=12,          # the ONNX version to export the model to
#                       do_constant_folding=True,  # whether to execute constant folding for optimization
#                     #   input_names = ['input'],   # the model's input names
#                     #   output_names = ['1250', '1295', '1294'],
#                     #   dynamic_axes={'input': {0: 'n', 1: 'c', 2: 'h', 3: 'w'}, # support dynamic reshape
#                     #                 'output': {0: 'n', 1: 'c', 2: 'h', 3: 'w'}}, # support dynamic reshape
#                       example_outputs=out) # the model's output names
# def save_model_tf2onnx():
#     pass
# def save_model_onnx2openvino():
#     pass
########################################################################################
# inference
class MlInferenceTools():
    def __init__(self, runtime_type, thread_num):
        self.runtime_type = runtime_type
        self.thread_num = thread_num
        if self.runtime_type == 0:
            # pytorch    
            self.resnet50_gpu = models.resnet50(pretrained=False, progress=False).to("cuda").eval()
            # print(self.resnet50_gpu)
            self.resnet50_gpu.load_state_dict(torch.load(weight_pth), strict=True)
            # BATCH_SIZE = 32
            # url='https://images.dog.ceo/breeds/retriever-golden/n02099601_3004.jpg'
            # img = resize(io.imread(url), (224, 224))
            # img = np.expand_dims(np.array(img, dtype=np.float32), axis=0) # Expand image to have a batch dimension
            # input_batch = np.array(np.repeat(img, BATCH_SIZE, axis=0), dtype=np.float32) # Repeat across the batch dimension
            # input_batch_chw = torch.from_numpy(input_batch).transpose(1,3).transpose(2,3)
            # self.input_batch_gpu = input_batch_chw.to("cuda")
            # print('input_batch_gpu.shape = ', self.input_batch_gpu.shape)

            
            # net = models.__dict__['resskspp']()
            # net = net.to(device)
            # net.load_state_dict(torch.load(pretrained_res_best, map_location=device))
            pass
        elif self.runtime_type == 1:
            # openvino
            self.net_vino = OpenVinoInfer(model_openvino, self.thread_num)
            pass
        elif  self.runtime_type == 2:
            # onnxruntime
            # save_model_torch2onnx()
            onnx_model = onnx.load(model_onnx)
            onnx.checker.check_model(onnx_model)
            sess_options = onnxruntime.SessionOptions()
            sess_options.intra_op_num_threads = self.thread_num
            sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.ort_session = onnxruntime.InferenceSession(model_onnx, sess_options)
        elif  self.runtime_type == 3:
            cv2.setNumThreads(self.thread_num)
            self.opencvdnn = OpenCvDnn(model_prototxt, model_caffemodel, 224, 224)
            pass
        elif  self.runtime_type == 4:
            f = open(model_trt, "rb")
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

            self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            self.h_input = cuda.pagelocked_empty(trt.volume(self.context.get_binding_shape(0)), dtype=np.float32)
            self.h_output = cuda.pagelocked_empty(trt.volume(self.context.get_binding_shape(1)), dtype=np.float32)
            # Allocate device memory for inputs and outputs.
            self.d_input = cuda.mem_alloc(self.h_input.nbytes)
            self.d_output = cuda.mem_alloc(self.h_output.nbytes)
            self.stream = cuda.Stream()
            pass
        elif  self.runtime_type == 5:

            dir_net="./caffe/deploy.prototxt"
            dir_weight="./caffe/model.caffemodel"
            self.net = caffe.Net(dir_net, dir_weight, caffe.TEST)
            

        else:
            pass

    def inference(self, inputs):
        if self.runtime_type == 0:
            # outputs, _ = net(inputs)
            with torch.no_grad():
                inputs = torch.from_numpy(inputs).cuda()
                predictions = np.array(self.resnet50_gpu(inputs).cpu())
                # predictions = np.array(self.resnet50_gpu(self.input_batch_gpu).cpu())
            # print('predictions.shape = ', predictions.shape)
            outputs = predictions
            # pass
        elif self.runtime_type == 1:
            outputs  = self.net_vino.forward(inputs)
        elif self.runtime_type == 2:
            # compute ONNX Runtime output prediction
            # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
            ort_inputs = {self.ort_session.get_inputs()[0].name: inputs}
            outputs = self.ort_session.run(None, ort_inputs)
        elif self.runtime_type == 3:
            outputs = self.opencvdnn(inputs)
            pass
        elif self.runtime_type == 4:

            def predict(batch): # result gets copied into output
                # transfer input data to device
                cuda.memcpy_htod_async(self.d_input, batch, self.stream)
                # execute model
                self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
                # transfer predictions back
                cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
                # syncronize threads
                self.stream.synchronize()
    
                return self.h_output

            # allocate device memory

            outputs = predict(inputs)
        elif self.runtime_type == 5:
            # 图片预处理 变更为模型中的尺寸
            transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
            transformer.set_transpose('data', (2,0,1))                            	#改变维度的顺序
            # transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
            # transformer.set_raw_scale('data', 255)                                # 缩放到【0，255】之间
            # 本身就是单通道
            # 如果是三通道模式下需要加上  transformer.set_channel_swap('data', (2,1,0))     
            # pycaffe加载图片函数 false表示单通道
            img = 'ch05003_20190711181206-00006820.jpg'
            im = caffe.io.load_image(img,True) 
            #执行上面设置的图片预处理操作，并将图片载入到blob中 
            # self.net.blobs['data'].data[...] = transformer.preprocess('data',im)   
            self.net.blobs['data'].data[...] = inputs  
            out = self.net.forward()
            # 将传播值转为数组
            outputs = out.get('fc_norm')
        else:
            pass
        return outputs
def val(device):
    # test img
    filename = "ch01001_20181110150903.mp4-134-0008208_0.858600.jpg"
    # img_np_nchw = get_img_np_nchw(filename, 224, 224)
    # img_np_nchw = get_img_np_nchw(filename, 108, 108)
    # img_np_nchw = img_np_nchw.astype(dtype=np.float32)

    # inference_input = img_np_nchw
    inference_input = im_list

    mit = MlInferenceTools(runtime_type, thread_num)
    # warm up 
    # nums_warm_up = 20
    nums_warm_up = 1
    for i in range(nums_warm_up):
        outputs = mit.inference(inference_input)
    # inference
    st = time.time()
    # nums = 300
    nums = 1
    for i in range(nums):
        outputs = mit.inference(inference_input)
    ed = time.time()
    average_time = (ed - st)/nums
    print("average_time = ",average_time)
    print("FPS = ",1 / average_time)
    # print(outputs)
    return outputs

if __name__=="__main__":
    res = val(device='cpu')
    f = open('res.txt',mode='w') 
    f.write(str(res))
