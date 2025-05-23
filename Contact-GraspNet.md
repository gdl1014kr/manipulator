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

## tf_sampling.cpp, tf_grouping.cpp, tf_interpolate.cpp 코드 수정

각각의 코드에서 return Status::OK(); 부분을 return ::tensorflow::OkStatus();로 수정
각각의 코드에서 .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
부분을 .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) -> ::tensorflow::Status { 로 수정

=>
이유:
1. TensorFlow 2.11+ 버전부터는 tsl::Status::OK()가 더 이상 제공되지 않고, Status() 또는 OkStatus()를 사용해야 함.
2. 람다 반환 타입 불일치
.SetShapeFn(...)에 전달하는 람다식은 명시적 반환 타입이 필요하거나, tensorflow::Status 타입으로 반환해야 함.


## Recompile pointnet2 tf_ops

sh compile_pointnet_tfops.sh



-------------------------
conda config --set channel_priority flexible


## Download Models & copy(https://drive.google.com/drive/folders/1tBHKf60K8DLM5arm-Chyf7jxkzOr5zGl 해당 링크에서 contact_graspnet_train_and_test-20250523T082241Z-1-001.zip 파일 다운)

cp ~/Downloads/contact_graspnet_train_and_test-20250523T082241Z-1-001.zip checkpoints/

## Download Test data(https://drive.google.com/drive/folders/1TqpM2wHAAo0j3i1neu3Xeru3_WnsYQnx 해당 링크에서 test_data-20250523T082527Z-1-001.zip 파일 다운)

mkdir test_data
cp ~/Downloads/test_data-20250523T082527Z-1-001.zip test_data/

## unzip

cd checkpoints/ && unzip contact_graspnet_train_and_test-20250523T082241Z-1-001.zip
cd ../test_data/   && unzip test_data-20250523T082527Z-1-001.zip
cd ..

## package install

conda install -c conda-forge opencv
conda install -c conda-forge trimesh
conda install -c conda-forge scipy
conda install -c conda-forge pyrender pyopengl trimesh pyglet
pip install pyrender
sudo apt-get update
sudo apt-get install -y libvtk7-dev python3-vtk7
conda install pyqt  
pip install mayavi




