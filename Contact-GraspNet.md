# Installation(
```bash
cd ~/ros2_nanoowl_ws/src
git clone https://github.com/NVlabs/contact_graspnet.git
cd contact_graspnet

## Create the conda env

conda create -n contact_graspnet python=3.10
conda activate contact_graspnet

## NVIDIA tensorflow GPU build install 

pip install --upgrade pip
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v60 \
    tensorflow==2.15.0+nv24.05
python -m pip install sympy
python -m pip install pillow
export TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
export TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')



## Recompile pointnet2 tf_ops
sh compile_pointnet_tfops.sh
