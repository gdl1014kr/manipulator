import argparse
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

def main(args):
    # 1. 모델 로드 (측정에서 제외되는 일회성 작업)
    print(f"--- Using Official Ultralytics Library with {args.model_type} file ---")
    print(f"Loading model from: {args.model_path}")
    # Ultralytics는 CUDA가 사용 가능하면 자동으로 GPU를 사용합니다.
    model = YOLO(args.model_path)
    print("Model loaded.")

    # 2. 클래스(프롬프트) 설정
    if args.model_type == 'pt':
        print(f"Setting classes to: {args.classes}")
        model.set_classes(args.classes)
    else:
        print("Note: set_classes() is skipped for ONNX/TensorRT models in this script.")
        print(f"Using baked-in classes: {model.names}")


    # 3. 이미지 로딩
    print(f"Loading image from: {args.image_path}")
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: Could not read image from {args.image_path}.")
        return

    # 4. 워밍업 (정확한 측정을 위해 GPU 예열)
    print(f"Warming up the model with {args.warmup} iterations...")
    for _ in range(args.warmup):
        # verbose=False로 설정하여 워밍업 중 로그 출력 방지
        model.predict(source=image, conf=0.35, imgsz=640, iou=0.7, verbose=False)
    
    # 파이토치에게 GPU의 모든 워밍업 연산이 끝날 때까지 기다리라고 명령
    torch.cuda.synchronize()
    print("Warmup finished.")

    # 5. 순수 추론 성능 측정 (반복 및 평균)
    print(f"Running inference measurement for {args.iters} iterations...")
    inference_latencies = []
    
    for _ in range(args.iters):
        # GPU 연산 시작 직전에 동기화
        torch.cuda.synchronize()
        start_time = time.time()

        # 추론 실행
        model.predict(source=image, conf=0.35, imgsz=640, iou=0.7, verbose=False)

        # GPU 연산 완료 직후에 동기화
        torch.cuda.synchronize()
        end_time = time.time()
        
        # 시간을 ms 단위로 변환하여 저장
        inference_latencies.append((end_time - start_time) * 1000)

    # 6. 결과 계산
    avg_inference_latency_ms = np.mean(inference_latencies)
    std_inference_latency_ms = np.std(inference_latencies)
    inference_fps = 1000.0 / avg_inference_latency_ms if avg_inference_latency_ms > 0 else 0

    # 7. 시각화를 위한 최종 추론 및 결과 이미지 생성
    print("Generating final image with detections...")
    results = model.predict(source=image, conf=0.35, imgsz=640, iou=0.7, verbose=False)
    result_img = results[0].plot() # Bounding box가 그려진 이미지

    # 8. 이미지에 성능 텍스트 추가
    perf_text_latency = f"Avg Latency: {avg_inference_latency_ms:.2f} ms"
    perf_text_fps = f"FPS: {inference_fps:.2f}"
    
    # 텍스트 위치, 폰트, 색상 등 설정
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_color = (255, 255, 255) # 흰색
    bg_color = (0, 0, 0) # 검은색 배경

    # 배경 사각형 추가 (가독성 향상)
    cv2.rectangle(result_img, (5, 5), (400, 80), bg_color, -1)
    
    # 텍스트 쓰기
    cv2.putText(result_img, perf_text_latency, (10, 40), font, font_scale, text_color, thickness)
    cv2.putText(result_img, perf_text_fps, (10, 70), font, font_scale, text_color, thickness)
    
    # 9. 최종 결과 출력 및 저장
    print("\n--- Performance Metrics (YOLO-World) ---")
    print(f"Model Type: {args.model_type.upper()}")
    print(f"Number of Iterations: {args.iters}")
    print(f"Average Inference Latency: {avg_inference_latency_ms:.2f} ms (+/- {std_inference_latency_ms:.2f} ms)")
    print(f"Inference FPS: {inference_fps:.2f}")

    out_path = "yolo_world_pt_result.png"
    cv2.imwrite(out_path, result_img)
    print(f"\nResult saved in {out_path}")
    
    cv2.imshow("YOLO-World Performance Test", result_img)
    print("Displaying result. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO-World Performance Measurement Script")
    parser.add_argument("--model_type", type=str, default="pt", choices=["pt", "onnx", "engine"], help="Type of the model file.")
    parser.add_argument("--model_path", type=str, default="yolov8x-worldv2.pt", help="Path to the model file.")
    parser.add_argument("--image_path", type=str, default="dogs.jpg", help="Path to the input image.")
    parser.add_argument("--classes", nargs='+', default=['a dog', 'person', 'car', 'cat'], help="List of classes to detect (for .pt model).")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations.")
    parser.add_argument("--iters", type=int, default=100, help="Number of measurement iterations.")
    args = parser.parse_args()
    main(args)
