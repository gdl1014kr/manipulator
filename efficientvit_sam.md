git clone https://github.com/mit-han-lab/efficientvit.git
cd efficientvit
           
conda create -n efficientvit python=3.10 -y
conda activate efficientvit
pip install -U -r requirements.txt

//https://github.com/mit-han-lab/efficientvit/blob/master/applications/efficientvit_sam/README.md 에서 EfficientViT-SAM-XLO Checkpoint Download//

mkdir -p assets/checkpoints/efficient_sam
mv ~/Downloads/efficientvit_sam_xl0.pt ~/ros2_nanoowl_ws/src/efficientvit/assets/checkpoints/efficientvit_sam/
