# nanosam tensorrt에서 image test & latency 측정하는 code(basic_usage_trt_latency.py)

# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import argparse
from nanosam.utils.predictor import Predictor
import time
import torch # <<< 수정된 부분: 동기화를 위해 torch 임포트

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_encoder", type=str, default="data/resnet18_image_encoder.engine")
    parser.add_argument("--mask_decoder", type=str, default="data/mobile_sam_mask_decoder.engine")
    parser.add_argument("--image", type=str, default="assets/dogs.jpg")
    # <<< 수정된 부분: 워밍업 및 반복 횟수 인자 추가 >>>
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations.")
    parser.add_argument("--iters", type=int, default=100, help="Number of measurement iterations.")
    args = parser.parse_args()
    
    # 1. 모델 로딩 (측정에서 제외되는 일회성 작업)
    print("Loading TensorRT engines for NanoSAM...")
    predictor = Predictor(
        args.image_encoder,
        args.mask_decoder
    )
    print("Engines loaded.")

    # 2. 이미지 로딩 (일회성 작업)
    print(f"Loading image: {args.image}")
    image = PIL.Image.open(args.image)
    
    # 추론에 사용할 프롬프트(바운딩 박스) 준비
    bbox = [100, 100, 850, 759]
    points = np.array([
        [bbox[0], bbox[1]],
        [bbox[2], bbox[3]]
    ])
    point_labels = np.array([2, 3])

    # 3. 워밍업 (정확한 측정을 위해 GPU 예열)
    print(f"Warming up the model with {args.warmup} iterations...")
    for _ in range(args.warmup):
        predictor.set_image(image)
        _, _, _ = predictor.predict(points, point_labels)
    
    torch.cuda.synchronize() # 워밍업 연산 완료 대기
    print("Warmup finished.")

    # 4. 순수 추론 성능 측정 (반복 및 평균)
    # NanoSAM은 set_image(인코더)와 predict(디코더)가 추론의 한 세트
    print(f"Running full inference measurement for {args.iters} iterations...")
    inference_latencies = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        start_time = time.time()

        # <<< 측정 대상: 인코더 + 디코더 전체 추론 >>>
        predictor.set_image(image)
        mask, _, _ = predictor.predict(points, point_labels)

        torch.cuda.synchronize()
        end_time = time.time()
        inference_latencies.append((end_time - start_time) * 1000) # ms로 저장

    # 5. 후처리 및 시각화 (일회성 작업)
    # 마지막 반복의 결과를 사용하여 후처리 및 시각화 진행
    mask_processed = (mask[0, 0] > 0).detach().cpu().numpy()

    # 결과 시각화
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.imshow(mask_processed, alpha=0.5, cmap='viridis')
    x = [bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]]
    y = [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]]
    plt.plot(x, y, 'g-')
    plt.axis('off')

    # 6. 결과 계산 및 출력
    avg_inference_latency_ms = np.mean(inference_latencies)
    std_inference_latency_ms = np.std(inference_latencies)
    inference_fps = 1000.0 / avg_inference_latency_ms if avg_inference_latency_ms > 0 else 0

    print("\n--- Performance Metrics (NanoSAM - TensorRT) ---")
    print(f"Number of Iterations: {args.iters}")
    print(f"Average Full Inference Latency: {avg_inference_latency_ms:.2f} ms (+/- {std_inference_latency_ms:.2f} ms)")
    print(f"Full Inference Theoretical FPS: {inference_fps:.2f}")

    # 이미지에 성능 텍스트 추가
    perf_text = (
        f"Avg Inference Latency (TRT): {avg_inference_latency_ms:.2f} ms\n"
        f"Inference FPS (TRT): {inference_fps:.2f}"
    )
    plt.text(20, 40, perf_text, color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.7))
    
    output_filename = "data/nanosam_trt_latency.jpg"
    plt.savefig(output_filename, bbox_inches="tight", pad_inches=0.0)
    print(f"\nDone. Result saved to {output_filename}")
