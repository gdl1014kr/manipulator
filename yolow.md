# yolow

yolow 찾아보니까 형용사도 인식 가능함.
다만, 최적화하기 전인 pt onnx 파일이나 tensorrt 로 변환할때는 코드로 target class(내장할 클래스(객체 지정)), 예를 들어 dog, car, person, cat 이런식으로 직접 
문제가 onnx나 tensorrt로 변환할때는 코드로 직접  target class(내장할 클래스(객체 지정)), 예를 들어 dog, car, person, cat 이런식으로 지정해줘야 한다는데 지정을 안하고 그냥 원래 방식대로 변환하면 계속 indexError, TypeError 같은게 뜸



## pt Download
https://docs.ultralytics.com/ko/models/yolo-world/#how-do-i-train-a-yolo-world-model-on-my-dataset

## ONNX Download
https://pypi.org/project/yolo-world-onnx/

## https://www.elinux.org/Jetson_Zoo#ONNX_Runtime 해당 링크에서 onnxruntime_gpu-1.18.0-cp310-cp310-linux_aarch64.whl 설치 
=> jetson은 GPU 기반이기 때문에 onnxruntime을 GPU용으로 설치해야함.

## onnx runtime(gpu) 설치: Downloads 폴더에서 pip3 install onnxruntime_gpu-1.18.0-cp310-cp310-linux_aarch64.whl 입력하여 gpu 버전의 onnx runtime 설치. 



A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash.
...
If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.

해당 오류 발생시

pip install "numpy<2.0"





## onnx -> tensorrt(engine) : fp16 (input 크기 1x3x640x640 고정인 static model 이므로 min/opt/max shape 명시 x)

trtexec --onnx=yolov8l-worldv2.onnx --saveEngine=yolov8l-worldv2.engine --fp16
trtexec --onnx=yolov8x-worldv2.onnx --saveEngine=yolov8x-worldv2.engine --fp16
trtexec --onnx=yolov8s-worldv2.onnx --saveEngine=yolov8s-worldv2.engine --fp16
trtexec --onnx=yolov8m-worldv2.onnx --saveEngine=yolov8m-worldv2.engine --fp16


pip install sentence-transformers
