# ai_inference_tools
1.docker:
docker run -ti --runtime=nvidia  --shm-size="8g"  -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/pytorch:20.03-py3

2.exec:
2.1 pytorch
python3 ai_inference_tools.py --runtime_type pytorch  --weights_dir xx.pth  --device cpu --input_synthetic
python3 ai_inference_tools.py --runtime_type pytorch  --weights_dir xx.pth  --device cpu --input_image_dir data/dog.jpg
2.2 caffe
python3 ai_inference_tools.py --runtime_type caffe  --net_dir xx.prototxt  --weights_dir xx.caffemodel  --device cpu