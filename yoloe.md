## pt Download
https://docs.ultralytics.com/ko/models/yolo-world/#how-do-i-train-a-yolo-world-model-on-my-dataset

## ONNX Download
https://pypi.org/project/yolo-world-onnx/

## pt -> onnx

yolo export model=yoloe-11l-seg.pt format=onnx 
yolo export model=yoloe-11m-seg.pt format=onnx 
yolo export model=yoloe-11s-seg.pt format=onnx 

## onnx -> tensorrt(engine) : fp16 (input 크기 1x3x640x640 고정인 static model 이므로 min/opt/max shape 명시 x)

sudo nvpmodel -m 0
sudo jetson_clocks
/usr/src/tensorrt/bin/trtexec --onnx=$HOME/ros2_nanoowl_ws/src/yoloe/onnx/yoloe-11l-seg.onnx --saveEngine=$HOME/ros2_nanoowl_ws/src/yoloe/tensorrt/yoloe-11l-seg.engine --fp16
/usr/src/tensorrt/bin/trtexec --onnx=$HOME/ros2_nanoowl_ws/src/yoloe/onnx/yoloe-11m-seg.onnx --saveEngine=$HOME/ros2_nanoowl_ws/src/yoloe/tensorrt/yoloe-11m-seg.engine --fp16
/usr/src/tensorrt/bin/trtexec --onnx=$HOME/ros2_nanoowl_ws/src/yoloe/onnx/yoloe-11s-seg.onnx --saveEngine=$HOME/ros2_nanoowl_ws/src/yoloe/tensorrt/yoloe-11s-seg.engine --fp16





