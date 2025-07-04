# nanosam tensorrt에서 image test & latency 측정하는 code(basic_usage_trt_latency.py)

# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# ... (라이선스 헤더는 동일) ...

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import argparse
from nanosam.utils.predictor import Predictor
import time # <<< 1. time 라이브러리 추가

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_encoder", type=str, default="data/resnet18_image_encoder.engine")
    parser.add_argument("--mask_decoder", type=str, default="data/mobile_sam_mask_decoder.engine")
    parser.add_argument("--image", type=str, default="assets/dogs.jpg") # <<< image 인자 추가
    args = parser.parse_args()
    
    # <<< 전체 파이프라인 시간 측정 시작 >>>
    total_start_time = time.time()
    
    # Instantiate TensorRT predictor
    print("Loading TensorRT engines...")
    predictor = Predictor(
        args.image_encoder,
        args.mask_decoder
    )
    print("Engines loaded.")

    # Read image
    print(f"Loading and preprocessing image: {args.image}")
    image = PIL.Image.open(args.image)
    
    # <<< 순수 AI 추론 시간 측정 시작 >>>
    inference_start_time = time.time()

    # Run image encoder
    predictor.set_image(image)

    # Segment using bounding box
    bbox = [100, 100, 850, 759]
    points = np.array([
        [bbox[0], bbox[1]],
        [bbox[2], bbox[3]]
    ])
    point_labels = np.array([2, 3])

    # Run mask decoder
    mask, _, _ = predictor.predict(points, point_labels)

    # <<< 순수 AI 추론 시간 측정 종료 >>>
    inference_end_time = time.time()

    mask = (mask[0, 0] > 0).detach().cpu().numpy()

    # <<< 전체 파이프라인 시간 측정 종료 >>>
    total_end_time = time.time()

    # <<< Latency 및 FPS 계산 >>>
    total_latency_ms = (total_end_time - total_start_time) * 1000
    inference_latency_ms = (inference_end_time - inference_start_time) * 1000
    total_fps = 1.0 / (total_latency_ms / 1000) if total_latency_ms > 0 else 0
    inference_fps = 1.0 / (inference_latency_ms / 1000) if inference_latency_ms > 0 else 0

    print("\n--- Performance Metrics (NanoSAM - TensorRT) ---")
    print(f"Total Pipeline Latency: {total_latency_ms:.2f} ms")
    print(f"Total Theoretical FPS: {total_fps:.2f}")
    print(f"Inference-Only Latency: {inference_latency_ms:.2f} ms")
    print(f"Inference-Only Theoretical FPS: {inference_fps:.2f}")

    # Draw results
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5, cmap='viridis') # 마스크 색상 변경
    x = [bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]]
    y = [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]]
    plt.plot(x, y, 'g-')
    
    # <<< 이미지에 성능 텍스트 추가 >>>
    perf_text = (
        f"Total Latency: {total_latency_ms:.2f} ms\nTotal FPS: {total_fps:.2f}\n"
        f"Inference Latency: {inference_latency_ms:.2f} ms\nInference FPS: {inference_fps:.2f}"
    )
    plt.text(20, 60, perf_text, color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.7))
    
    plt.axis('off')
    
    output_filename = "data/trt_perf_test_out.jpg"
    plt.savefig(output_filename)
    print(f"\nDone. Result saved to {output_filename}")
