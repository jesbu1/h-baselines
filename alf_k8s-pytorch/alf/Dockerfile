FROM docker.hobot.cc/dlp/tf:runtime-py3.6-cudnn7.4-cuda10.0-ubuntu18.04

MAINTAINER jesse.zhang@horizon.ai

# replace source list
COPY ./sources.list /etc/apt/sources.list

# basic software package
RUN apt update && apt install -y \
    vim emacs tree wget unzip \
    git cmake \
    libxml2 libxml2-dev libxslt1-dev \
    dirmngr gnupg2 lsb-release \
    xvfb kmod swig patchelf \
    libopenmpi-dev  libcups2-dev  libssl-dev  libosmesa6-dev \
    python3-pip

# install gazebo-9
RUN echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > \
    /etc/apt/sources.list.d/gazebo-stable.list

RUN wget http://packages.osrfoundation.org/gazebo.key -O - |  apt-key add -

RUN apt update && apt-get install -q -y \
    mesa-utils \
    gazebo9 \
    libgazebo9-dev


RUN apt install -y ffmpeg

# install nvidia driver
RUN wget 'http://cn.download.nvidia.com/XFree86/Linux-x86_64/410.93/NVIDIA-Linux-x86_64-410.93.run' \
    && chmod +x NVIDIA-Linux-x86_64-410.93.run \
    && ./NVIDIA-Linux-x86_64-410.93.run -a -s --no-kernel-module  --no-x-check \
    && rm -f NVIDIA-Linux-x86_64-410.93.run


# install python requirments
RUN mkdir -p /root/.pip/
COPY ./pip.conf /root/.pip/pip.conf
RUN pip3 install --upgrade pip

COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt \
    && rm -rf /tmp/requirements.txt

## prepare mujoco library
COPY ./package /tmp/package
#RUN mv /tmp/package/mujoco/udevadm /bin/udevadm &&  chmod +x /bin/udevadm
RUN wget https://www.roboti.us/download/mujoco200_linux.zip \
    && unzip mujoco200_linux.zip -d /opt/.mujoco \
    && rm -f mujoco200_linux.zip

RUN mv /tmp/package/mujoco/mjkey.txt /opt/.mujoco/
ENV MUJOCO_PY_MUJOCO_PATH=/opt/.mujoco/mujoco200_linux
ENV MUJOCO_PY_MJKEY_PATH=/opt/.mujoco/mjkey.txt
ENV LD_LIBRARY_PATH=/opt/.mujoco/mujoco200_linux/bin:${LD_LIBRARY_PATH}
RUN pip3 install lockfile
RUN pip3 install mujoco-py==2.0.2.0 --root=/opt/
ENV PYTHONPATH=/opt/usr/local/lib/python3.6/dist-packages:${PYTHONPATH}
RUN chmod -R a+w /opt/usr/
RUN python3 -c "import mujoco_py"

# clean
RUN  rm -rf /var/lib/apt/lists/* \
     && rm -rf /tmp/package \
     && rm -rf /install \
     && rm -rf /home

COPY ./set_env.sh /opt/set_env.sh
