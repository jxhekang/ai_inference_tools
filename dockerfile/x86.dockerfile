# According to https://github.com/ultralytics/yolov5/blob/v5.0/Dockerfile

# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:21.03-py3

# Install linux packages

# Install python dependencies
# COPY requirements.txt .
# RUN python -m pip install --upgrade pip
# RUN python -m pip install --upgrade pip --force-reinstall -i https://pypi.tuna.tsinghua.edu.cn/simple/ 

# RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
# RUN pip install --no-cache -r requirements.txt coremltools onnx gsutil notebook

# Set environment variables
# ENV HOME=/usr/src/app
# ARG DEBIAN_FRONTEND=noninteractive

# ARG TEMP_DIR=/tmp/
# WORKDIR ${TEMP_DIR}
# COPY 3rd ./

