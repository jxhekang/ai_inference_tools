import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from termcolor import cprint

class TensorRtInfer:
    def __init__(self, args):
        cprint('TensorRtInfer init', 'green')
        f = open(args.net_dir, "rb")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

        self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.h_input = cuda.pagelocked_empty(trt.volume(self.context.get_binding_shape(0)), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume(self.context.get_binding_shape(1)), dtype=np.float32)
        # Allocate device memory for inputs and outputs.
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.stream = cuda.Stream()

        # self.input_shape = xx
        # self.output_shape = xx

    def forward(self, input):
        # transfer input data to device
        cuda.memcpy_htod_async(self.d_input, input, self.stream)
        # execute model
        self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
        # transfer predictions back
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        # syncronize threads
        self.stream.synchronize()
    
        return self.h_output
