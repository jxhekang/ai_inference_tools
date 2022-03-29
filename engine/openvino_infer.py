import os
import logging as log# from openvino.inference_engine import IECore
from openvino.inference_engine import IECore

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
        
        # self.input_shape = xx
        # self.output_shape = xx

    def forward(self, x):
        # Start sync inference
        log.info("Starting inference in synchronous mode")
        res = self.exec_net.infer(inputs={self.input_blob: x})
        return res