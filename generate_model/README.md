python3 generate_model.py --arch resnet50 --model_type onnx
python3 generate_model.py --arch resnet50 --model_type torchscript
python3 generate_model.py --arch mobilenet_v2 --model_type onnx
python3 generate_model.py --arch mobilenet_v2 --model_type torchscript
python3 generate_model.py --arch MLP --self_define True --model_type onnx
python3 generate_model.py --arch MLP --self_define True --model_type torchscript

python3 generate_model.py --arch EagleEyeFace3 --self_define True --model_type torchscript

导出失败，torchscript导出onnx不行啊。。。
python3 generate_model.py --arch pelee_pelee_20200810_epoch_19_516000 --self_define True --model_type onnx