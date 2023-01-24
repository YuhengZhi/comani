# Built by referencing https://github.com/selkies-project/docker-nvidia-egl-desktop
FROM nvidia/cudagl:11.2.2-devel-ubuntu20.04

ARG NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES all
ENV TZ UTC
ENV VGL_DISPLAY egl
ARG VIRTUALGL_VERSION=3.0.2

RUN apt-get update
RUN apt-get install -y python3 python3-dev python3-pip libglib2.0-0 libsm6 libxrender1 libxext6\
    vim wget python3-setuptools curl

RUN curl -fsSL -O "https://sourceforge.net/projects/virtualgl/files/virtualgl_${VIRTUALGL_VERSION}_amd64.deb" && \
    curl -fsSL -O "https://sourceforge.net/projects/virtualgl/files/virtualgl32_${VIRTUALGL_VERSION}_amd64.deb" && \
    apt-get update && apt-get install -y --no-install-recommends ./virtualgl_${VIRTUALGL_VERSION}_amd64.deb ./virtualgl32_${VIRTUALGL_VERSION}_amd64.deb && \
    rm -f "virtualgl_${VIRTUALGL_VERSION}_amd64.deb" "virtualgl32_${VIRTUALGL_VERSION}_amd64.deb" && \
    rm -rf /var/lib/apt/lists/* && \
    chmod u+s /usr/lib/libvglfaker.so && \
    chmod u+s /usr/lib/libdlfaker.so && \
    chmod u+s /usr/lib32/libvglfaker.so && \
    chmod u+s /usr/lib32/libdlfaker.so && \
    chmod u+s /usr/lib/i386-linux-gnu/libvglfaker.so && \
    chmod u+s /usr/lib/i386-linux-gnu/libdlfaker.so

COPY requirements.txt .

RUN pip install -r requirements.txt