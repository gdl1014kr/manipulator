# nanosam onnx에서 image test & latency 측정하는 code(basic_usage_onnx_latency.py)

import argparse
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import onnxruntime as ort
import torch
import torch.nn.functional as F
from copy import deepcopy
import time # <<< 1. time 라이브러리 추가

# (SamResize, preprocess_image 등 모든 보조 함수는 이전과 동일)
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
    mask_torch = torch.from_numpy(mask)
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
    args = parser.parse_args()

    # <<< 전체 파이프라인 시간 측정 시작 >>>
    total_start_time = time.time()

    # 1. ONNX 모델 로딩
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
    
    # 2. 이미지 로딩 및 전처리
    print(f"Loading and preprocessing image: {args.image}")
    image = PIL.Image.open(args.image)
    image_np = np.array(image)
    image_tensor_np = preprocess_image(image_np, 1024)

    # <<< 순수 AI 추론 시간 측정 시작 >>>
    inference_start_time = time.time()

    # 3. 이미지 인코딩 실행
    print("Running encoder...")
    encoder_input_name = encoder_session.get_inputs()[0].name
    encoder_output_name = encoder_session.get_outputs()[0].name
    features = encoder_session.run([encoder_output_name], {encoder_input_name: image_tensor_np})[0]

    # 4. 디코더 실행을 위한 프롬프트(바운딩 박스) 준비
    bbox = [100, 100, 850, 759]
    points = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32)
    point_labels = np.array([2, 3], dtype=np.float32)

    # 5. 디코더 실행
    print("Running decoder...")
    decoder_input_feed = {
        "image_embeddings": features,
        "point_coords": np.array([points]),
        "point_labels": np.array([point_labels]),
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input": np.zeros(1, dtype=np.float32)
    }
    decoder_output_names = ["iou_predictions", "low_res_masks"]
    _, low_res_masks = decoder_session.run(decoder_output_names, decoder_input_feed)

    # <<< 순수 AI 추론 시간 측정 종료 >>>
    inference_end_time = time.time()

    # 6. 결과 후처리 및 시각화
    print("Postprocessing and saving result...")
    hi_res_mask = upscale_mask(low_res_masks, (image.height, image.width))
    mask = (hi_res_mask[0, 0] > 0).cpu().numpy()

    # <<< 전체 파이프라인 시간 측정 종료 >>>
    total_end_time = time.time()

    # <<< Latency 및 FPS 계산 >>>
    total_latency_ms = (total_end_time - total_start_time) * 1000
    inference_latency_ms = (inference_end_time - inference_start_time) * 1000
    total_fps = 1.0 / (total_latency_ms / 1000) if total_latency_ms > 0 else 0
    inference_fps = 1.0 / (inference_latency_ms / 1000) if inference_latency_ms > 0 else 0
    
    print("\n--- Performance Metrics (ONNX Runtime) ---")
    print(f"Total Pipeline Latency: {total_latency_ms:.2f} ms")
    print(f"Total Theoretical FPS: {total_fps:.2f}")
    print(f"Inference-Only Latency: {inference_latency_ms:.2f} ms")
    print(f"Inference-Only Theoretical FPS: {inference_fps:.2f}")

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_box(bbox, plt.gca())
    plt.axis('off')
    
    # <<< 이미지에 성능 텍스트 추가 >>>
    perf_text = (
        f"Total Latency: {total_latency_ms:.2f} ms\nTotal FPS: {total_fps:.2f}\n"
        f"Inference Latency: {inference_latency_ms:.2f} ms\nInference FPS: {inference_fps:.2f}"
    )
    plt.text(20, 60, perf_text, color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.7))
    
    output_filename = "data/nanosam_onnx_perf_test_out.jpg"
    plt.savefig(output_filename, bbox_inches="tight", pad_inches=0.0)
    print(f"\nDone. Result saved to {output_filename}")
