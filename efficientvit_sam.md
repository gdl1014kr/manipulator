git clone https://github.com/mit-han-lab/efficientvit.git
cd efficientvit/applications/efficientvit_sam
           
conda create -n efficientvit python=3.10 -y
conda activate efficientvit
pip install -U -r requirements.txt


mkdir -p assets/checkpoints/efficientvit_sam
wget -O assets/checkpoints/efficientvit_sam/efficientvit_sam_xl0.pt \
  https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_xl0.pt

sudo apt update
sudo apt install nano -y
