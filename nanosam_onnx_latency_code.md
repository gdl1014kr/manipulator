## https://www.elinux.org/Jetson_Zoo#ONNX_Runtime 해당 링크에서 onnxruntime_gpu-1.18.0-cp310-cp310-linux_aarch64.whl 설치 
=> jetson은 GPU 기반이기 때문에 onnxruntime을 GPU용으로 설치해야함.

## onnx runtime(gpu) 설치: Downloads 폴더에서 pip3 install onnxruntime_gpu-1.18.0-cp310-cp310-linux_aarch64.whl 입력하여 gpu 버전의 onnx runtime 설치. 

## 설치한 onnxruntime 버전에 맞게 numpy 다운그레이드
pip install "numpy<2"





-----------------------------------------------------------------------------------

# nanosam onnx에서 image test & latency 측정하는 code(basic_usage_onnx_latency.py)

import argparse
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import onnxruntime as ort
import torch
import torch.nn.functional as F
from copy import deepcopy
import time

# (SamResize, show_mask, show_box 등 모든 보조 함수는 이전과 동일하게 유지)
class SamResize:
    def __init__(self, size: int):
        self.size = size
    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        long_side = max(h, w)
        if long_side != self.size:
            return self.apply_image(image)
        else:
            return image
    def apply_image(self, image: np.ndarray) -> np.ndarray:
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.size)
        return np.array(PIL.Image.fromarray(image).resize((target_size[1], target_size[0]), PIL.Image.BILINEAR))
    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> tuple[int, int]:
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

def preprocess_image(image: np.ndarray, size: int = 1024) -> np.ndarray:
    image_mean = np.array([123.675, 116.28, 103.53])
    image_std = np.array([58.395, 57.12, 57.375])
    resize_transform = SamResize(size)
    image = resize_transform(image)
    image = (image.astype(np.float32) - image_mean) / image_std
    image = image.transpose(2, 0, 1) # HWC -> CHW
    h, w = image.shape[-2:]
    pad_h = size - h
    pad_w = size - w
    image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), 'constant', constant_values=0)
    return np.expand_dims(image, axis=0).astype(np.float32)

def upscale_mask(mask: np.ndarray, image_shape: tuple, size: int = 256) -> torch.Tensor:
    # GPU에서 후처리를 위해 torch 텐서로 변환하고 cuda로 보냄
    mask_torch = torch.from_numpy(mask).cuda()
    if image_shape[1] > image_shape[0]: # width > height
        lim_x = size
        lim_y = int(size * image_shape[0] / image_shape[1])
    else:
        lim_x = int(size * image_shape[1] / image_shape[0])
        lim_y = size
    mask_torch = mask_torch[:, :, :lim_y, :lim_x]
    image_shape_torch = torch.Size(image_shape)
    hi_res_mask = F.interpolate(mask_torch, size=image_shape_torch, mode='bilinear', align_corners=False)
    return hi_res_mask

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NanoSAM inference with ONNX models and measure performance.")
    parser.add_argument("--image_encoder", type=str, required=True, help="Path to the ONNX image encoder model.")
    parser.add_argument("--mask_decoder", type=str, required=True, help="Path to the ONNX mask decoder model.")
    parser.add_argument("--image", type=str, default="assets/dogs.jpg", help="Path to the input image.")
    # <<< 수정된 부분: 워밍업 및 반복 횟수 인자 추가 >>>
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations.")
    parser.add_argument("--iters", type=int, default=100, help="Number of measurement iterations.")
    args = parser.parse_args()

    # 1. ONNX 모델 로딩 (일회성 작업)
    print("Loading ONNX models...")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        encoder_session = ort.InferenceSession(args.image_encoder, providers=providers)
        decoder_session = ort.InferenceSession(args.mask_decoder, providers=providers)
        print(f"Encoder is using: {encoder_session.get_providers()}")
        print(f"Decoder is using: {decoder_session.get_providers()}")
    except Exception as e:
        print(f"Error loading ONNX models: {e}")
        exit()
    print("Models loaded.")
    
    # 2. 이미지 로딩 및 전처리 (일회성 작업)
    print(f"Loading and preprocessing image: {args.image}")
    image = PIL.Image.open(args.image)
    image_np = np.array(image)
    image_tensor_np = preprocess_image(image_np, 1024)
    
    # 추론에 사용할 프롬프트(바운딩 박스) 및 입력 준비
    bbox = [100, 100, 850, 759]
    points = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32)
    point_labels = np.array([2, 3], dtype=np.float32)
    encoder_input_name = encoder_session.get_inputs()[0].name
    encoder_output_name = encoder_session.get_outputs()[0].name
    decoder_output_names = ["iou_predictions", "low_res_masks"]
    
    # 3. 워밍업 (정확한 측정을 위해 GPU 예열)
    print(f"Warming up the ONNX model with {args.warmup} iterations...")
    for _ in range(args.warmup):
        features = encoder_session.run([encoder_output_name], {encoder_input_name: image_tensor_np})[0]
        decoder_input_feed = { "image_embeddings": features, "point_coords": np.array([points]), "point_labels": np.array([point_labels]), "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32), "has_mask_input": np.zeros(1, dtype=np.float32) }
        _ = decoder_session.run(decoder_output_names, decoder_input_feed)
        
    torch.cuda.synchronize() # 워밍업 연산 완료 대기
    print("Warmup finished.")

    # 4. 순수 추론 성능 측정 (반복 및 평균)
    print(f"Running full ONNX inference measurement for {args.iters} iterations...")
    inference_latencies = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        start_time = time.time()

        # <<< 측정 대상: 인코더 + 디코더 전체 추론 >>>
        features = encoder_session.run([encoder_output_name], {encoder_input_name: image_tensor_np})[0]
        decoder_input_feed = { "image_embeddings": features, "point_coords": np.array([points]), "point_labels": np.array([point_labels]), "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32), "has_mask_input": np.zeros(1, dtype=np.float32) }
        _, low_res_masks = decoder_session.run(decoder_output_names, decoder_input_feed)

        torch.cuda.synchronize()
        end_time = time.time()
        inference_latencies.append((end_time - start_time) * 1000)

    # 5. 후처리 및 시각화 (일회성 작업)
    print("Postprocessing and saving result...")
    hi_res_mask = upscale_mask(low_res_masks, (image.height, image.width))
    mask = (hi_res_mask[0, 0] > 0).cpu().numpy()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_box(bbox, plt.gca())
    plt.axis('off')
    
    # 6. 결과 계산 및 출력
    avg_inference_latency_ms = np.mean(inference_latencies)
    std_inference_latency_ms = np.std(inference_latencies)
    inference_fps = 1000.0 / avg_inference_latency_ms if avg_inference_latency_ms > 0 else 0
    
    print("\n--- Performance Metrics (NanoSAM - ONNX Runtime) ---")
    print(f"Number of Iterations: {args.iters}")
    print(f"Average Full Inference Latency: {avg_inference_latency_ms:.2f} ms (+/- {std_inference_latency_ms:.2f} ms)")
    print(f"Full Inference Theoretical FPS: {inference_fps:.2f}")

    # 이미지에 성능 텍스트 추가
    perf_text = (
        f"Avg Inference Latency (ONNX): {avg_inference_latency_ms:.2f} ms\n"
        f"Inference FPS (ONNX): {inference_fps:.2f}"
    )
    plt.text(20, 40, perf_text, color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.7))
    
    output_filename = "data/nanosam_onnx_latency.jpg"
    plt.savefig(output_filename, bbox_inches="tight", pad_inches=0.0)
    print(f"\nDone. Result saved to {output_filename}")
