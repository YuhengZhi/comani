FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
LABEL Name=comani Version=0.0.1

WORKDIR /comani

COPY . .

ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
# ENV MESA_GL_VERSION_OVERRIDE=3.3

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update && apt-get install -y libgl1-mesa-glx libosmesa6 libglfw3-dev libgles2-mesa-dev libxtst6 libxv1 libglu1-mesa libegl1-mesa dpkg mesa-utils-extra
RUN dpkg -i virtualgl_3.0.1_amd64.deb

RUN pip install --no-cache-dir -r requirements.txt