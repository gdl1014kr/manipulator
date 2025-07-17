## pt Download
https://docs.ultralytics.com/ko/models/yolo-world/#how-do-i-train-a-yolo-world-model-on-my-dataset

## ONNX Download
https://pypi.org/project/yolo-world-onnx/


## onnx -> tensorrt(engine) : fp16 (input 크기 1x3x640x640 고정인 static model 이므로 min/opt/max shape 명시 x)

trtexec --onnx=yolov8l-worldv2.onnx --saveEngine=yolov8l-worldv2.engine --fp16
trtexec --onnx=yolov8x-worldv2.onnx --saveEngine=yolov8x-worldv2.engine --fp16
trtexec --onnx=yolov8s-worldv2.onnx --saveEngine=yolov8s-worldv2.engine --fp16
trtexec --onnx=yolov8m-worldv2.onnx --saveEngine=yolov8m-worldv2.engine --fp16
