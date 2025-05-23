# terminator install

sudo apt install terminator

# Hangul install

sudo apt upgrade ibus-hangul -y

------------------------------------------------------------------------------------------------------
# Chromium browser install

sudo snap install chromium
----------------------------------------------------------------------------------------------------
# Firefox install

sudo snap install firefox
---------------------------------------------------------------------------------------------------
# ROS2 Humble install

## Set locale(UTF-8)

locale  # check for UTF-8

sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

locale  # verify settings


## Universe repository is enabled

sudo apt install software-properties-common
sudo add-apt-repository universe

##  ROS 2 GPG key with apt

sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

## package update & system upgrade

sudo apt update
sudo apt upgrade

## ROS2 Humble install

sudo apt install ros-humble-desktop


## ROS-Base install

sudo apt install ros-humble-ros-base

## Development tools install

sudo apt install ros-dev-tools


-------------------------------------------------------------------------------------------------------------
# NanoOWL Setup(Install the dependencies)

## 1. Pytorch & torchvision install- Jetpack 6.0(https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html 사이트 참고) 

### Install system packages required by PyTorch

sudo apt-get -y update; 
sudo apt-get install -y  python3-pip libopenblas-dev;

### 환경변수에 whl 파일 경로 지정(https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048 해당 링크에서 torch 2.3.0 download 선행)

export TORCH_INSTALL=/home/iram/Downloads/torch-2.3.0-cp310-cp310-linux_aarch64.whl

### PATH 환경변수 설정

echo 'export PATH=$PATH:/home/iram/.local/bin' >> ~/.bashrc
source ~/.bashrc

### onnx install

python3 -m pip install onnx==1.14.1

### Pytorch install

python3 -m pip install --upgrade pip; python3 -m pip install numpy==1.26.1; python3 -m pip install --no-cache $TORCH_INSTALL

### torchvision install(https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048 해당 링크에서 torchvision 0.18.0 download 선행)

python3 -m pip install --no-cache-dir \
  ~/Downloads/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl


## 2. torch2trt install

### install the torch2trt Python library
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install --user

## 3. Install NVIDIA TensorRT(jetpack sdk 설치할때 자동 설치=> 생략)

## 4. Install the Transformers library

python3 -m pip install transformers


# Install the NanoOWL package.

git clone https://github.com/NVIDIA-AI-IOT/nanoowl
cd nanoowl
python3 setup.py develop --user

# Build the TensorRT engine for the OWL-ViT vision encoder

mkdir -p data
python3 -m nanoowl.build_image_encoder_engine \
    data/owl_image_encoder_patch32.engine

# Run an example prediction to ensure everything is working

cd examples
python3 owl_predict.py \
    --prompt="[an owl, a glove]" \
    --threshold=0.1 \
    --image_encoder_engine=../data/owl_image_encoder_patch32.engine
---------------------------------------------------------------------------------------------
# ROS2-NanoOWL Setup

## 원하는 위치에 ROS2NanoOWL 워크스페이스 생성
mkdir -p ~/ROS2NanoOWL/src
cd ~/ROS2NanoOWL/src

## Git 리포지토리 클론

cd ~/ROS2NanoOWL/src
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
git clone https://github.com/NVIDIA-AI-IOT/ROS2-NanoOWL.git
git clone https://github.com/NVIDIA-AI-IOT/nanoowl.git
git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
git clone --branch humble https://github.com/ros2/demos.git

## Docker 컨테이너 실행

sudo usermod -aG docker $USER
newgrp docker
cd ~/ROS2NanoOWL/src/isaac_ros_common
./scripts/run_dev.sh -d ~/ROS2NanoOWL

##
cd ..
git clone --branch v0.18.0 https://github.com/pytorch/vision.git
cd vision
pip install .




---------------------------------------------------------------------------------------------
# NanoSAM Setup(Install the dependencies)

## (optional) Install TRTPose - For the pose example.

cd ~
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose
python3 setup.py develop --user

## (optional) Install the Transformers library - For the OWL ViT example.

cd ~
cd nanosam
python3 -m pip install transformers


# Install the NanoSAM Python package

cd ~
git clone https://github.com/NVIDIA-AI-IOT/nanosam
cd nanosam
python3 setup.py develop --user

# Build the TensorRT engine for the mask decoder

## 1. install timm
python3 -m pip install --user timm

## 2. Export the MobileSAM mask decoder ONNX file
mkdir -p data
python3 -m nanosam.tools.export_sam_mask_decoder_onnx \
    --model-type=vit_t \
    --checkpoint=assets/mobile_sam.pt \
    --output=data/mobile_sam_mask_decoder.onnx

## 3. PATH에 trtexec 경로 추가
echo 'export PATH=$PATH:/usr/src/tensorrt/bin' >> ~/.bashrc
source ~/.bashrc


-----------------------------------------

# KGN 

OS: Ubuntu 22.04 (JetPack 6.0)

Python: 3.10 (conda env kgn)

PyTorch: v2.3.0 (CUDA 12.2 런타임 포함)

TorchVision: v0.18

CUDA Toolkit: 12.2

1. Installation

## Miniconda install - Jetson AGX Orin(JetPack 6.0, Ubuntu 22.04 aarch64)에서는 x86_64용 Anaconda 설치 스크립트 실행 시 illegal instruction 오류가 빈번히 발생하여
공식적으로는 경량화된 Miniconda 설치를 권장

### 설치 스크립트 다운로드

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh \
  -O ~/Miniconda3-latest-Linux-aarch64.sh

### 스크립트 실행 권한 부여 및 설치 시작
chmod +x ~/Miniconda3-latest-Linux-aarch64.sh
bash ~/Miniconda3-latest-Linux-aarch64.sh

### 설치 완료 후 셸 초기화
source ~/miniconda3/bin/activate
conda init

## Miniconda 환경 생성 및 관리

### 새 환경 생성
conda create -n kgn python=3.10

### 환경 활성화
conda activate kgn

## torchaudio install(https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048 해당 링크에서 torchviaudio download 선행)

python3 -m pip install --no-cache-dir ~/Downloads/torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl


## DCNv2 clone & install

export DCNv2_ROOT=~/ros2_nanoowl_ws/src/DCNv2_latest
git clone https://github.com/jinfagang/DCNv2_latest $DCNv2_ROOT
cd $DCNv2_ROOT
python setup.py install --user


## KGN clone

cd ..
export KGN_ROOT=~/ros2_nanoowl_ws/src/KGN
git clone https://github.com/ivalab/KGN.git $KGN_ROOT 
cd $KGN_ROOT
pip install -r requirements.txt 



2. Demo

# ros2_nanoowl_ws/src/KGN/exp 폴더에 kgnv2.pth 넣음. 
mkdir exp

# ros2_nanoowl_ws/src/KGN/src/lib/utils 의 keypoints.py 299번째 줄 return ori_cls.astype(np.int)를 return ori_cls.astype(int)로 수정

bash experiments/demo_kgnv2.sh
