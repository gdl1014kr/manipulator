# hangul install

sudo apt upgrade ibus-hangul -y


# Chromium browser install

## snap

sudo snap install chromium

# Firefox install

## snap

sudo snap install firefox

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

# NanoOWL Setup(Install the dependencies)

## 1. Pytorch & torchvision install- Jetpack 6.0(https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html 사이트 참고) 

### Install system packages required by PyTorch

sudo apt-get -y update; 
sudo apt-get install -y  python3-pip libopenblas-dev;

### 환경변수에 whl 파일 경로 지정(https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048 해당 링크에서 torch 2.2.0 download 선행)

export TORCH_INSTALL=/home/iram/Downloads/torch-2.3.0-cp310-cp310-linux_aarch64.whl

### PATH 환경변수 설정

echo 'export PATH=$PATH:/home/iram/.local/bin' >> ~/.bashrc
source ~/.bashrc

### onnx install

python3 -m pip install onnx==1.14.1

### Pytorch install

python3 -m pip install --upgrade pip; python3 -m pip install numpy==1.26.1; python3 -m pip install --no-cache $TORCH_INSTALL

### torchvision install



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
