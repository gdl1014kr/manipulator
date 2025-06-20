cd ros2_nanoowl_ws/src
git clone https://github.com/mit-han-lab/efficientvit.git
cd efficientvit
           
conda create -n efficientvit python=3.10 -y
conda activate efficientvit

## 1. Pytorch & torchvision install- Jetpack 6.0(https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html 사이트 참고) 

### Install system packages required by PyTorch

sudo apt-get -y update; 
sudo apt-get install -y  python3-pip libopenblas-dev;

### 환경변수에 whl 파일 경로 지정(https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048 해당 링크에서 torch 2.3.0 download 선행)

export TORCH_INSTALL=/home/iram/Downloads/torch-2.3.0-cp310-cp310-linux_aarch64.whl

### PATH 환경변수 설정

echo 'export PATH=$PATH:/home/iram/.local/bin' >> ~/.bashrc
source ~/.bashrc

### Pytorch install

python3 -m pip install --upgrade pip; python3 -m pip install numpy==1.26.1; python3 -m pip install --no-cache $TORCH_INSTALL

### torchvision install(https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048 해당 링크에서 torchvision 0.18.0 download 선행)

python3 -m pip install --no-cache-dir \
  ~/Downloads/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl

## torch, torchvision 주석 처리 or 삭제 -> ctrl+x, Y, Enter 저장 및 종료
nano requirements.txt 

## Segment Anything 원본 라이브러리 설치
pip install git+https://github.com/facebookresearch/segment-anything.git

## 나머지 패키지 설치
pip install -r requirements.txt


mkdir -p assets/checkpoints/efficientvit_sam
wget -O assets/checkpoints/efficientvit_sam/efficientvit_sam_xl0.pt \
  https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_xl0.pt

sudo apt update
sudo apt install nano -y


## 모델 실행 및 사용
cd applications/efficientvit_sam

