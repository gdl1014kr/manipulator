cd vlm/src
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

## vlm/src/efficientvit/efficientvit/models/nn/norm.py를 아래와 같이 수정(triton 오류 방지)

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

# <<< 수정된 부분 시작 / Modified section start
try:
    from efficientvit.models.nn.triton_rms_norm import TritonRMSNorm2dFunc
except ImportError:
    TritonRMSNorm2dFunc = None
# <<< 수정된 부분 끝 / Modified section end

from efficientvit.models.utils import build_kwargs_from_config

__all__ = ["LayerNorm2d", "TritonRMSNorm2d", "build_norm", "reset_bn", "set_norm_eps"]


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


class TritonRMSNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #
        # This part will not be executed if Triton is not available,
        # because the code that calls this class (`build_norm`) will not
        # be able to build it if TritonRMSNorm2dFunc is None.
        # However, to be extra safe, we can add a check here.
        #
        if TritonRMSNorm2dFunc is None:
            raise RuntimeError("Triton is not available, TritonRMSNorm2d cannot be used.")
        return TritonRMSNorm2dFunc.apply(x, self.weight, self.bias, self.eps)


# register normalization function here
REGISTERED_NORM_DICT: dict[str, type] = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
    "ln2d": LayerNorm2d,
    "trms2d": TritonRMSNorm2d,
}


def build_norm(name="bn2d", num_features=None, **kwargs) -> Optional[nn.Module]:
    if name in ["ln", "ln2d", "trms2d"]:
        #
        # This check prevents building TritonRMSNorm2d if Triton is not available.
        #
        if name == "trms2d" and TritonRMSNorm2dFunc is None:
            # Fallback to a different norm or return None if Triton is not available
            print("Warning: Triton not available. Falling back from trms2d to ln2d.")
            name = "ln2d"
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features

    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None


def reset_bn(
    model: nn.Module,
    data_loader: list,
    sync=True,
    progress_bar=False,
) -> None:
    import copy

    import torch.nn.functional as F
    from tqdm import tqdm

    from efficientvit.apps.utils import AverageMeter, is_master, sync_tensor
    from efficientvit.models.utils import get_device, list_join

    bn_mean = {}
    bn_var = {}

    tmp_model = copy.deepcopy(model)
    for name, m in tmp_model.named_modules():
        if isinstance(m, _BatchNorm):
            bn_mean[name] = AverageMeter(is_distributed=False)
            bn_var[name] = AverageMeter(is_distributed=False)

            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    x = x.contiguous()
                    if sync:
                        batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
                        batch_mean = sync_tensor(batch_mean, reduce="cat")
                        batch_mean = torch.mean(batch_mean, dim=0, keepdim=True)

                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
                        batch_var = sync_tensor(batch_var, reduce="cat")
                        batch_var = torch.mean(batch_var, dim=0, keepdim=True)
                    else:
                        batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)

                    batch_mean = torch.squeeze(batch_mean)
                    batch_var = torch.squeeze(batch_var)

                    mean_est.update(batch_mean.data, x.size(0))
                    var_est.update(batch_var.data, x.size(0))

                    # bn forward using calculated mean & var
                    _feature_dim = batch_mean.shape[0]
                    return F.batch_norm(
                        x,
                        batch_mean,
                        batch_var,
                        bn.weight[:_feature_dim],
                        bn.bias[:_feature_dim],
                        False,
                        0.0,
                        bn.eps,
                    )

                return lambda_forward

            m.forward = new_forward(m, bn_mean[name], bn_var[name])

    # skip if there is no batch normalization layers in the network
    if len(bn_mean) == 0:
        return

    tmp_model.eval()
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="reset bn", disable=not progress_bar or not is_master()) as t:
            for images in data_loader:
                images = images.to(get_device(tmp_model))
                tmp_model(images)
                t.set_postfix(
                    {
                        "bs": images.size(0),
                        "res": list_join(images.shape[-2:], "x"),
                    }
                )
                t.update()

    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, _BatchNorm)
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)


def set_norm_eps(model: nn.Module, eps: Optional[float] = None) -> None:
    for m in model.modules():
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm, _BatchNorm)):
            if eps is not None:
                m.eps = eps

## vlm/src/efficientvit/efficientvit/models/nn/__init__.py 에서 from.triton_rms_norm import *에 주석 표시(triton 오류 방지)

## TensorRT
cd ~/vlm/src/efficientvit/

mkdir -p assets/export_models/efficientvit_sam/onnx/
mkdir -p assets/export_models/efficientvit_sam/tensorrt/

###  l1_encoder.onnx 이동
mv ~/Downloads/l1_encoder.onnx ~/vlm/src/efficientvit/assets/export_models/efficientvit_sam/onnx/
### l1_decoder.onnx 이동
mv ~/Downloads/l1_decoder.onnx ~/vlm/src/efficientvit/assets/export_models/efficientvit_sam/onnx/
### XL0_encoder.onnx 이동
mv ~/Downloads/xl0_encoder.onnx ~/vlm/src/efficientvit/assets/export_models/efficientvit_sam/onnx/
### XL0_decoder.onnx 이동
mv ~/Downloads/xl0_decoder.onnx ~/vlm/src/efficientvit/assets/export_models/efficientvit_sam/onnx/

### ONNX 파일을 TensorRT로 변환(l1, fp16)

#### l1 Encoder
/usr/src/tensorrt/bin/trtexec \
    --onnx=assets/export_models/efficientvit_sam/onnx/l1_encoder.onnx \
    --minShapes=input_image:1x3x512x512 \
    --optShapes=input_image:1x3x512x512 \
    --maxShapes=input_image:1x3x512x512 \
    --saveEngine=assets/export_models/efficientvit_sam/tensorrt/l1_encoder_fp16.engine \
    --fp16

#### l1 Encoder(GPU)
/usr/src/tensorrt/bin/trtexec \
    --onnx=assets/export_models/efficientvit_sam/onnx/l1_encoder.onnx \
    --minShapes=input_image:1x3x512x512 \
    --optShapes=input_image:1x3x512x512 \
    --maxShapes=input_image:1x3x512x512 \
    --saveEngine=assets/export_models/efficientvit_sam/tensorrt/l1_encoder_fp16_native.engine \
    --fp16 \
    --layerDeviceTypes=*:GPU

#### l1 Decoder
/usr/src/tensorrt/bin/trtexec \
    --onnx=assets/export_models/efficientvit_sam/onnx/l1_decoder.onnx \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:1x5x2,point_labels:1x5 \
    --maxShapes=point_coords:1x10x2,point_labels:1x10 \
    --saveEngine=assets/export_models/efficientvit_sam/tensorrt/l1_decoder_fp16.engine \
    --fp16

#### l1 Decoder(GPU)
/usr/src/tensorrt/bin/trtexec \
    --onnx=assets/export_models/efficientvit_sam/onnx/l1_decoder.onnx \
    --saveEngine=assets/export_models/efficientvit_sam/tensorrt/l1_decoder_fp16_native.engine \
    --fp16 \
    --layerDeviceTypes=*:GPU \
    --minShapes=image_embeddings:1x256x64x64,point_coords:1x1x2,point_labels:1x1 \
    --optShapes=image_embeddings:1x256x64x64,point_coords:1x5x2,point_labels:1x5 \
    --maxShapes=image_embeddings:1x256x64x64,point_coords:1x10x2,point_labels:1x10

### ONNX 파일을 TensorRT로 변환(l1, int8)

#### l1 Encoder
/usr/src/tensorrt/bin/trtexec \
    --onnx=assets/export_models/efficientvit_sam/onnx/l1_encoder.onnx \
    --minShapes=input_image:1x3x512x512 \
    --optShapes=input_image:1x3x512x512 \
    --maxShapes=input_image:1x3x512x512 \
    --saveEngine=assets/export_models/efficientvit_sam/tensorrt/l1_encoder_int8.engine \
    --int8

#### l1 Decoder
/usr/src/tensorrt/bin/trtexec \
    --onnx=assets/export_models/efficientvit_sam/onnx/l1_decoder.onnx \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:1x5x2,point_labels:1x5 \
    --maxShapes=point_coords:1x10x2,point_labels:1x10 \
    --saveEngine=assets/export_models/efficientvit_sam/tensorrt/l1_decoder_int8.engine \
    --int8


### ONNX 파일을 TensorRT로 변환(XL0, fp16)

#### XL0 Encoder
trtexec --onnx=assets/export_models/efficientvit_sam/onnx/xl0_encoder.onnx --minShapes=input_image:1x3x1024x1024 --optShapes=input_image:4x3x1024x1024 --maxShapes=input_image:4x3x1024x1024 --saveEngine=assets/export_models/efficientvit_sam/tensorrt/efficientvit_sam_xl0_encoder.engine


#### XL0 Decoder

trtexec --onnx=assets/export_models/efficientvit_sam/onnx/xl0_decoder.onnx --minShapes=point_coords:1x1x2,point_labels:1x1 --optShapes=point_coords:16x2x2,point_labels:16x2 --maxShapes=point_coords:16x2x2,point_labels:16x2 --fp16 --saveEngine=assets/export_models/efficientvit_sam/tensorrt/efficientvit_sam_xl0_decoder.engine



### ONNX 파일을 TensorRT로 변환(XL0, int8)

#### XL0 Encoder
/usr/src/tensorrt/bin/trtexec \
    --onnx=assets/export_models/efficientvit_sam/onnx/xl0_encoder.onnx \
    --minShapes=input_image:1x3x1024x1024 \
    --optShapes=input_image:1x3x1024x1024 \
    --maxShapes=input_image:1x3x1024x1024 \
    --saveEngine=assets/export_models/efficientvit_sam/tensorrt/xl0_encoder_int8.engine \
    --int8

#### XL0 Decoder
/usr/src/tensorrt/bin/trtexec \
    --onnx=assets/export_models/efficientvit_sam/onnx/xl0_decoder.onnx \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:1x5x2,point_labels:1x5 \
    --maxShapes=point_coords:1x10x2,point_labels:1x10 \
    --saveEngine=assets/export_models/efficientvit_sam/tensorrt/xl0_decoder_int8.engine \
    --int8

### Tensorrt python package를 conda 환경으로 연결(Symbolic link)

#### tensorrt package 실제 위치 확인
ls -ld /usr/lib/python3.10/dist-packages/tensorrt

#### 가상 환경에 Symbolic link 생성
ln -s /usr/lib/python3.10/dist-packages/tensorrt $(python -c 'import site; print(site.getsitepackages()[0])')/tensorrt

#### link 생성 확인
ls -l $(python -c 'import site; print(site.getsitepackages()[0])') | grep tensorrt


### L1(512x512) TensorRT image test(point)
python applications/efficientvit_sam/run_efficientvit_sam_trt.py \
    --model efficientvit-sam-l1 \
    --encoder_engine assets/export_models/efficientvit_sam/tensorrt/l1_encoder_fp16.engine \
    --decoder_engine assets/export_models/efficientvit_sam/tensorrt/l1_decoder_fp16.engine \
    --img_path example.png \
    --mode point

### L1(512x512) TensorRT image test(boxes)
python applications/efficientvit_sam/run_efficientvit_sam_trt.py \
    --model efficientvit-sam-l1 \
    --encoder_engine assets/export_models/efficientvit_sam/tensorrt/l1_encoder_fp16.engine \
    --decoder_engine assets/export_models/efficientvit_sam/tensorrt/l1_decoder_fp16.engine \
    --img_path example.png \
    --mode boxes \
    --boxes "[[200, 150, 450, 400]]"



### XL0(1024x1024) TensorRT image test(point)
python applications/efficientvit_sam/run_efficientvit_sam_trt.py \
    --model efficientvit-sam-xl0 \
    --encoder_engine assets/export_models/efficientvit_sam/tensorrt/xl0_encoder_fp16.engine \
    --decoder_engine assets/export_models/efficientvit_sam/tensorrt/xl0_decoder_fp16.engine \
    --img_path example.png \
    --mode point


### tensorrt(engine) 파일 latency 출력 & image test(l1):
python applications/efficientvit_sam/efficientvit_trt_latency.py --model efficientvit-sam-l1 --encoder_engine assets/export_models/efficientvit_sam/tensorrt/l1_encoder_fp16.engine --decoder_engine assets/export_models/efficientvit_sam/tensorrt/l1_decoder_fp16.engine --img_path example.png --mode point

### tensorrt(engine) 파일 latency 출력 & image test(xl0):
python applications/efficientvit_sam/efficientvit_trt_latency.py --model efficientvit-sam-xl0 --encoder_engine assets/export_models/efficientvit_sam/tensorrt/xl0_encoder_fp16.engine --decoder_engine assets/export_models/efficientvit_sam/tensorrt/xl0_decoder_fp16.engine --img_path example.png --mode point

### onnx 파일 image test:
python applications/efficientvit_sam/run_efficientvit_sam_onnx.py --model efficientvit-sam-l1 --encoder_model assets/export_models/efficientvit_sam/onnx/l1_encoder.onnx --decoder_model assets/export_models/efficientvit_sam/onnx/l1_decoder.onnx --mode point

### onnx 파일 latency 출력 & image test(l1):
python applications/efficientvit_sam/efficientvit_onnx_latency.py --model efficientvit-sam-l1 --encoder_model assets/export_models/efficientvit_sam/onnx/l1_encoder.onnx --decoder_model assets/export_models/efficientvit_sam/onnx/l1_decoder.onnx --mode point

### onnx 파일 latency 출력 & image test(xl0):
python applications/efficientvit_sam/efficientvit_onnx_latency.py --model efficientvit-sam-xl0 --encoder_model assets/export_models/efficientvit_sam/onnx/xl0_encoder.onnx --decoder_model assets/export_models/efficientvit_sam/onnx/xl0_decoder.onnx --mode point

-----------------------------------------------------------------------------------------
## intel RealSense Camera SDK 설치
pip install pyrealsense2
