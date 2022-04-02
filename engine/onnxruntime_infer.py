
import onnx
import onnxruntime
from termcolor import cprint

class OnnxRuntimeInfer:
    def __init__(self, model_dir, thread_num):
        onnx_model = onnx.load(model_dir)
        onnx.checker.check_model(onnx_model)
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = thread_num
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_session = onnxruntime.InferenceSession(model_dir, sess_options)
        # self.input_shape = xx
        # self.output_shape = xx
        
    def forward(self, input):
        ort_inputs = {self.ort_session.get_inputs()[0].name: input}
        outputs = self.ort_session.run(None, ort_inputs)
        return outputs
