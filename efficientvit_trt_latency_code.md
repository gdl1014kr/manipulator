# efficientvit_trt_experiment.py(latency & image test)

import argparse
import os
from copy import deepcopy
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from torch2trt import TRTModule
from torchvision.transforms.functional import resize


class SamResize:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        h, w, _ = image.shape
        long_side = max(h, w)
        if long_side != self.size:
            return self.apply_image(image)
        else:
            return image.permute(2, 0, 1)

    def apply_image(self, image: torch.Tensor) -> torch.Tensor:
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.size)
        return resize(image.permute(2, 0, 1), target_size)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> tuple[int, int]:
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size})"


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )
    ax.scatter(
        neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )


def preprocess(x, img_size, device):
    pixel_mean = [123.675 / 255, 116.28 / 255, 103.53 / 255]
    pixel_std = [58.395 / 255, 57.12 / 255, 57.375 / 255]

    x = torch.tensor(x).to(device)
    resize_transform = SamResize(img_size)
    x = resize_transform(x).float() / 255
    x = transforms.Normalize(mean=pixel_mean, std=pixel_std)(x)

    h, w = x.shape[-2:]
    th, tw = img_size, img_size
    assert th >= h and tw >= w
    x = F.pad(x, (0, tw - w, 0, th - h), value=0).unsqueeze(0)

    return x


def resize_longest_image_size(input_image_size: torch.Tensor, longest_side: int) -> torch.Tensor:
    input_image_size = input_image_size.to(torch.float32)
    scale = longest_side / torch.max(input_image_size)
    transformed_size = scale * input_image_size
    transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
    return transformed_size


def mask_postprocessing(masks: torch.Tensor, orig_im_size: torch.Tensor) -> torch.Tensor:
    img_size = 1024
    masks = masks.clone().detach()
    orig_im_size = torch.tensor(orig_im_size, device='cuda')

    masks = F.interpolate(
        masks,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )

    prepadded_size = resize_longest_image_size(orig_im_size, img_size)
    masks = masks[..., : int(prepadded_size[0]), : int(prepadded_size[1])]
    
    h, w = orig_im_size[0], orig_im_size[1]
    masks = F.interpolate(masks, size=(h.item(), w.item()), mode="bilinear", align_corners=False)
    return masks


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


def apply_coords(coords, original_size, new_size):
    old_h, old_w = original_size
    new_h, new_w = new_size
    coords = deepcopy(coords).astype(float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords


def apply_boxes(boxes, original_size, new_size):
    boxes = apply_coords(boxes.reshape(-1, 2, 2), original_size, new_size)
    return boxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model type.")
    parser.add_argument("--encoder_engine", type=str, required=True, help="TRT engine.")
    parser.add_argument("--decoder_engine", type=str, required=True, help="TRT engine.")
    parser.add_argument("--img_path", type=str, default="assets/example.png")
    parser.add_argument("--out_path", type=str, default=".demo/efficientvit_sam_demo_tensorrt.png")
    parser.add_argument("--mode", type=str, default="point", choices=["point", "boxes"])
    parser.add_argument("--point", type=str, default=None)
    parser.add_argument("--boxes", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations.")
    parser.add_argument("--iters", type=int, default=100, help="Number of measurement iterations.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    
    # 1. 모델 로딩 (측정에서 제외되는 일회성 작업)
    print("Loading TensorRT engines...")
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(args.encoder_engine, "rb") as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
    trt_encoder = TRTModule(engine, input_names=["input_image"], output_names=["image_embeddings"])

    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(args.decoder_engine, "rb") as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
    trt_decoder = TRTModule(
        engine,
        input_names=["image_embeddings", "point_coords", "point_labels"],
        output_names=["masks", "iou_predictions"],
    )
    print("Engines loaded.")

    # 2. 이미지 로딩 및 전처리
    raw_img = cv2.cvtColor(cv2.imread(args.img_path), cv2.COLOR_BGR2RGB)
    origin_image_size = raw_img.shape[:2]

    if args.model in ["efficientvit-sam-l0", "efficientvit-sam-l1", "efficientvit-sam-l2"]:
        img = preprocess(raw_img, img_size=512, device="cuda")
    elif args.model in ["efficientvit-sam-xl0", "efficientvit-sam-xl1"]:
        img = preprocess(raw_img, img_size=1024, device="cuda")
    else:
        raise NotImplementedError

    # 3. 추론에 사용할 프롬프트 준비
    input_size = get_preprocess_shape(*origin_image_size, long_side_length=1024)
    if args.mode == "point":
        H, W, _ = raw_img.shape
        point = np.array(yaml.safe_load(f"[[[{W // 2}, {H // 2}, {1}]]]" if args.point is None else args.point), dtype=np.float32)
        point_coords = point[..., :2]
        point_labels = point[..., 2]
        orig_point_coords = deepcopy(point_coords)
        orig_point_labels = deepcopy(point_labels)
        point_coords_tensor = torch.from_numpy(apply_coords(point_coords, origin_image_size, input_size).astype(np.float32)).to("cuda")
        point_labels_tensor = torch.from_numpy(point_labels).to("cuda")

    elif args.mode == "boxes":
        boxes = np.array(yaml.safe_load(args.boxes), dtype=np.float32)
        orig_boxes = deepcopy(boxes)
        point_coords_tensor = torch.from_numpy(apply_boxes(boxes, origin_image_size, input_size).astype(np.float32)).to("cuda")
        point_labels_tensor = torch.from_numpy(np.array([[2, 3] for _ in range(boxes.shape[0])], dtype=np.float32).reshape((-1, 2))).to("cuda")
    else:
        raise NotImplementedError
        
    # 4. 워밍업 (정확한 측정을 위해 GPU 예열)
    print(f"Warming up the model with {args.warmup} iterations...")
    for _ in range(args.warmup):
        image_embedding = trt_encoder(img)
        image_embedding = image_embedding[0].reshape(1, 256, 64, 64)
        inputs = (image_embedding, point_coords_tensor, point_labels_tensor)
        _ = trt_decoder(*inputs)
    
    torch.cuda.synchronize() # 워밍업 연산 완료 대기
    print("Warmup finished.")

    # 5. 순수 추론 성능 측정 (반복 및 평균)
    print(f"Running full inference measurement for {args.iters} iterations...")
    inference_latencies = []
    
    for i in range(args.iters):
        torch.cuda.synchronize()
        start_time = time.time()

        # <<< 수정된 부분: 인코더와 디코더 모두 반복문 안에서 측정 >>>
        image_embedding = trt_encoder(img)
        image_embedding = image_embedding[0].reshape(1, 256, 64, 64)
        inputs = (image_embedding, point_coords_tensor, point_labels_tensor)
        low_res_masks, _ = trt_decoder(*inputs)

        torch.cuda.synchronize()
        end_time = time.time()
        inference_latencies.append((end_time - start_time) * 1000) # ms로 저장

    # 6. 후처리 및 시각화 (일회성 작업)
    low_res_masks = low_res_masks.reshape(1, -1, 256, 256)
    masks = mask_postprocessing(low_res_masks, origin_image_size)[0]
    masks = masks > 0.0
    masks = masks.cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(raw_img)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=len(masks) > 1)
    
    if args.mode == "point":
        show_points(orig_point_coords, orig_point_labels, plt.gca())
    elif args.mode == "boxes":
        for box in orig_boxes:
            show_box(box, plt.gca())
            
    plt.axis("off")

    # 7. 결과 계산 및 출력
    avg_inference_latency_ms = np.mean(inference_latencies)
    std_inference_latency_ms = np.std(inference_latencies)
    inference_fps = 1000.0 / avg_inference_latency_ms if avg_inference_latency_ms > 0 else 0

    # <<< 수정된 부분: 출력 메시지를 'Full Inference'로 명확화 >>>
    print("\n--- Performance Metrics (EfficientViT - TensorRT) ---")
    print(f"Number of Iterations: {args.iters}")
    print(f"Average Full Inference Latency: {avg_inference_latency_ms:.2f} ms (+/- {std_inference_latency_ms:.2f} ms)")
    print(f"Full Inference Theoretical FPS: {inference_fps:.2f}")

    # 이미지에 성능 텍스트 추가
    perf_text = (
        f"Avg Full Inference Latency (TRT): {avg_inference_latency_ms:.2f} ms\n"
        f"Full Inference FPS (TRT): {inference_fps:.2f}"
    )
    plt.text(20, 40, perf_text, color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.7))

    plt.savefig(args.out_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
    print(f"\nResult saved in {args.out_path}")
