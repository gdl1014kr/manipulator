git clone https://github.com/mit-han-lab/efficientvit.git
cd efficientvit

python3 -m pip install torch torchvision timm           # PyTorch 및 필수 라이브러리  
python3 -m pip install torch2trt                        # TensorRT 연동  
python3 -m pip install transformers                     # OWL-ViT 사용  

//https://github.com/mit-han-lab/efficientvit/blob/master/applications/efficientvit_sam/README.md 에서 EfficientViT-SAM-XLO Checkpoint Download//

