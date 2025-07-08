Jetson agx orin: YOLOE(Object detection과 segmentation이 모두 됨.)
Jetson orin nano: 
Grounding dino(fp16, int8) + NanoSAM(Encoder: fp16, decoder: fp16, int8)
Grounding dino(fp16, int8) + EfficientVIT(Encoder: fp16, decoder: fp16, int8)

경우의 수 총 8가지:
1. Grounding dino(fp16) + NanoSAM(Encoder: fp16, decoder: fp16)
2. Grounding dino(fp16) + NanoSAM(Encoder: fp16, decoder: int8)
3. Grounding dino(int8) + NanoSAM(Encoder: fp16, decoder: fp16)
4. Grounding dino(int8) + NanoSAM(Encoder: fp16, decoder: int8)

5. Grounding dino(fp16) + EfficientVIT(Encoder: fp16, decoder: fp16)
6. Grounding dino(fp16) + EfficientVIT(Encoder: fp16, decoder: int8)
7. Grounding dino(int8) + EfficientVIT(Encoder: fp16, decoder: fp16)
8. Grounding dino(int8) + EfficientVIT(Encoder: fp16, decoder: int8)

=> 각 최적화 버전 별로 fps(latency) & test image 출력


1. segmentation 먼저 진행해서, 각 알고리즘 별로 결과 사진과 latency 뽑기(순수 추론 시간- encoder & decoder)
2. grounding dino latency 뽑기(순수 추론 시간- encoder & decoder)
3. 그리고 합쳐서 latency & 결과 사진 뽑기(전체 파이프라인- 이미지 파일 읽기 -> 전처리 -> encoder 추론 -> decoder 추론 -> 후처리 -> 결과 이미지 저장)

그리고 efficient는 weight 파일 2개로 다 실험해보기

dino : onnx encoder + onnx decoder, tensorrt (fp16+int8, fp16+fp16)

nanosam : onnx encoder + onnx decoder, tensorrt (fp16+int8, fp16+fp16)

efficient small : onnx encoder + onnx decoder, tensorrt (fp16+int8, fp16+fp16)

efficient large : onnx encoder + onnx decoder, tensorrt (fp16+int8, fp16+fp16, int8+fp16)


efficientvit에서 tensorrt(engine) 파일  image test:
python applications/efficientvit_sam/efficientvit_trt_latency.py --model efficientvit-sam-xl0 --encoder_engine assets/export_models/efficientvit_sam/tensorrt/xl0_encoder_fp16.engine --decoder_engine assets/export_models/efficientvit_sam/tensorrt/xl0_decoder_fp16.engine --mode point


efficientvit에서 tensorrt(engine) 파일 latency 출력 & image test:
python applications/efficientvit_sam/efficientvit_trt_latency.py --model efficientvit-sam-xl0 --encoder_engine assets/export_models/efficientvit_sam/tensorrt/xl0_encoder_fp16.engine --decoder_engine assets/export_models/efficientvit_sam/tensorrt/xl0_decoder_fp16.engine --mode point



efficientvit에서 onnx 파일 image test:
python applications/efficientvit_sam/run_efficientvit_sam_onnx.py --model efficientvit-sam-l1 --encoder_model assets/export_models/efficientvit_sam/onnx/l1_encoder.onnx --decoder_model assets/export_models/efficientvit_sam/onnx/l1_decoder.onnx --mode point

efficientvit에서 onnx 파일 latency 출력 & image test(l1):
python applications/efficientvit_sam/efficientvit_onnx_latency.py --model efficientvit-sam-l1 --encoder_model assets/export_models/efficientvit_sam/onnx/l1_encoder.onnx --decoder_model assets/export_models/efficientvit_sam/onnx/l1_decoder.onnx --mode point

efficientvit에서 onnx 파일 latency 출력 & image test(xl0):
python applications/efficientvit_sam/efficientvit_onnx_latency.py --model efficientvit-sam-xl0 --encoder_model assets/export_models/efficientvit_sam/onnx/xl0_encoder.onnx --decoder_model assets/export_models/efficientvit_sam/onnx/xl0_decoder.onnx --mode point



nanosam에서 tensorrt(engine)파일 image test
python3 examples/basic_usage.py --image_encoder=data/image_encoder_fp16_origin.engine --mask_decoder=data/mask_decoder_origin.engine

nanosam에서 tensorrt(engine)파일 latency 출력 & image test
python3 examples/basic_usage_trt_latency.py --image_encoder=data/image_encoder_fp16_origin.engine --mask_decoder=data/mask_decoder_fp16_origin.engine


nanosam에서 onnx 파일 image test
python3 examples/basic_usage_onnx.py --image_encoder="data/image_encoder.onnx" --mask_decoder="data/mask_decoder.onnx"

nanosam에서 onnx 파일 latency 출력 & image test
python3 examples/basic_usage_onnx_latency.py --image_encoder="data/image_encoder.onnx" --mask_decoder="data/mask_decoder.onnx"


sudo jetson_clocks

