import argparse
import os
import time
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import tensorrt as trt
from torch2trt import TRTModule
from torchvision.transforms.functional import resize as F_resize
from ultralytics import YOLO

# --- EfficientViT-SAM Helper Functions (Start) ---

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
        return F_resize(image.permute(2, 0, 1), target_size)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> tuple[int, int]:
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

def preprocess_image_for_sam(x: np.ndarray, img_size: int, device: str) -> torch.Tensor:
    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).to(device)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).to(device)
    
    x = torch.from_numpy(x).to(device)
    resize_transform = SamResize(img_size)
    x = resize_transform(x)
    x = (x - pixel_mean.view(-1, 1, 1)) / pixel_std.view(-1, 1, 1)

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

def postprocess_masks(masks: torch.Tensor, original_size: tuple[int, int], resized_size: tuple[int, int]) -> torch.Tensor:
    masks = F.interpolate(
        masks,
        size=resized_size,
        mode="bilinear",
        align_corners=False,
    )
    masks = masks[..., : resized_size[0], : resized_size[1]]
    masks = F.interpolate(masks, size=original_size, mode="bilinear", align_corners=False)
    return masks

def apply_coords(coords, original_size, new_size):
    old_h, old_w = original_size
    new_h, new_w = new_size
    coords = deepcopy(coords).astype(float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords

# --- EfficientViT-SAM Helper Functions (End) ---


def main(args):
    # --- ✨ 최종 수정: argparse로 받은 모델 타입에 따라 동적으로 경로 설정 ---
    model_type = args.sam_model_type
    base_path = "../efficientvit/assets/export_models/efficientvit_sam/tensorrt"
    sam_encoder_path = os.path.join(base_path, f"{model_type}_encoder_fp16.engine")
    sam_decoder_path = os.path.join(base_path, f"{model_type}_decoder_fp16.engine")

    # 1. 모델 로딩
    print("--- Loading All Models ---")
    print(f"Loading YOLO-World model from: {args.yolo_model}")
    yolo_model = YOLO(args.yolo_model)
    yolo_model.to('cuda')
    
    print(f"Loading EfficientViT-SAM models: Encoder='{sam_encoder_path}', Decoder='{sam_decoder_path}'")
    if not os.path.exists(sam_encoder_path) or not os.path.exists(sam_decoder_path):
        print(f"Error: Engine files not found for model type '{model_type}'.")
        print(f"Checked paths: \n- {sam_encoder_path}\n- {sam_decoder_path}")
        return

    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(sam_encoder_path, "rb") as f:
            engine_bytes = f.read()
        encoder_engine = runtime.deserialize_cuda_engine(engine_bytes)
    
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(sam_decoder_path, "rb") as f:
            engine_bytes = f.read()
        decoder_engine = runtime.deserialize_cuda_engine(engine_bytes)
        
    sam_encoder = TRTModule(encoder_engine, input_names=["input_image"], output_names=["image_embeddings"])
    sam_decoder = TRTModule(decoder_engine, input_names=["image_embeddings", "point_coords", "point_labels"], output_names=["masks", "iou_predictions"])
    print("All models loaded.")
    
    # --- ✨ 최종 수정: 모델 타입에 따라 이미지 크기 자동 설정 ---
    img_size_for_sam = 1024 if model_type in ["xl0", "xl1"] else 512

    # 2. 워밍업
    print(f"\nWarming up the full pipeline with {args.warmup} iterations...")
    image_for_warmup = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    
    for _ in range(args.warmup):
        yolo_model.set_classes(args.text_prompt)
        results = yolo_model.predict(image_for_warmup, verbose=False)
        
        if len(results[0].boxes) > 0:
            sam_img_tensor = preprocess_image_for_sam(image_for_warmup, img_size=img_size_for_sam, device="cuda")
            image_embedding = sam_encoder(sam_img_tensor)
            image_embedding = image_embedding.reshape(1, 256, 64, 64)
            
            for box_obj in results[0].boxes:
                box = box_obj.xyxy[0].cpu().numpy()
                input_size = SamResize.get_preprocess_shape(image_for_warmup.shape[0], image_for_warmup.shape[1], img_size_for_sam)
                sam_box = apply_coords(np.array([box.reshape(2, 2)]), image_for_warmup.shape[:2], input_size)
                point_coords = torch.from_numpy(sam_box).to("cuda", dtype=torch.float32).reshape(1, 2, 2)
                point_labels = torch.tensor([[2, 3]], dtype=torch.float32, device="cuda")
                sam_decoder(image_embedding, point_coords, point_labels)

    torch.cuda.synchronize()
    print("Warmup finished.")

    # 3. 전체 파이프라인 성능 측정
    print(f"\nRunning End-to-End pipeline measurement for {args.iters} iterations...")
    pipeline_latencies = []
    
    for _ in range(args.iters):
        torch.cuda.synchronize()
        start_time = time.time()
        
        image_bgr = cv2.imread(args.image)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_size = image_rgb.shape[:2]

        yolo_model.set_classes(args.text_prompt)
        results = yolo_model.predict(image_rgb, verbose=False)

        if len(results[0].boxes) == 0:
            print(f"Warning: No objects detected for prompts '{args.text_prompt}'. Skipping.")
            continue
        
        sam_img_tensor = preprocess_image_for_sam(image_rgb, img_size=img_size_for_sam, device="cuda")
        image_embedding = sam_encoder(sam_img_tensor)
        image_embedding = image_embedding.reshape(1, 256, 64, 64)

        overlay_img = image_bgr.copy()

        for box_obj in results[0].boxes:
            box = box_obj.xyxy[0].cpu().numpy()
            
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            point = np.array([[x_center, y_center]])

            input_size = SamResize.get_preprocess_shape(original_size[0], original_size[1], img_size_for_sam)
            sam_point = apply_coords(point, original_size, input_size)
            
            point_coords = torch.from_numpy(sam_point).to("cuda", dtype=torch.float32).unsqueeze(0)
            point_labels = torch.tensor([[1]], dtype=torch.float32, device="cuda")

            low_res_masks, iou_predictions = sam_decoder(image_embedding, point_coords, point_labels)
            best_mask_idx = torch.argmax(iou_predictions[0])
            selected_mask = low_res_masks[0, best_mask_idx, :, :].unsqueeze(0).unsqueeze(0)
            
            masks = postprocess_masks(selected_mask, original_size, input_size)[0]
            mask_processed = (masks > 0.0).cpu().numpy()[0]
            
            color = np.array([0, 255, 0], dtype=np.uint8)
            alpha = 0.5
            
            overlay_img[mask_processed] = (overlay_img[mask_processed] * (1 - alpha) + color * alpha).astype(np.uint8)
            
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (0, 0, 255), 3)

            conf = box_obj.conf[0].item()
            cls_id = int(box_obj.cls[0].item())
            prompt_text = args.text_prompt[cls_id]
            label = f"{prompt_text}: {conf:.2f}"
            cv2.putText(overlay_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imwrite(args.out_path, overlay_img)
        
        torch.cuda.synchronize()
        end_time = time.time()
        pipeline_latencies.append((end_time - start_time) * 1000)

    if not pipeline_latencies:
        print("Could not measure performance.")
        return

    # 4. 최종 결과 계산 및 출력
    avg_latency = np.mean(pipeline_latencies)
    std_latency = np.std(pipeline_latencies)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0
    
    final_img = cv2.imread(args.out_path)
    perf_text_latency = f"Avg Pipeline Latency: {avg_latency:.2f} ms"
    perf_text_fps = f"Pipeline FPS: {fps:.2f}"
    
    cv2.rectangle(final_img, (5, 5), (500, 80), (0, 0, 0), -1)
    cv2.putText(final_img, perf_text_latency, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(final_img, perf_text_fps, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite(args.out_path, final_img)
    
    print("\n--- Performance Metrics (YOLO-World + EfficientViT-SAM Pipeline) ---")
    print(f"Average End-to-End Pipeline Latency: {avg_latency:.2f} ms (+/- {std_latency:.2f} ms)")
    print(f"End-to-End Pipeline FPS: {fps:.2f}")
    print(f"\nResult saved in {args.out_path}")
    
    cv2.imshow("YOLO-World + EfficientViT-SAM Pipeline", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO-World + EfficientViT-SAM Pipeline")
    parser.add_argument("--yolo_model", type=str, default="yolov8s-worldv2.pt")
    # --- ✨ 최종 수정: 기본 모델을 l1로 변경하고, 선택지를 주석으로 명시 ---
    parser.add_argument("--sam_model_type", type=str, default="l1", help="EfficientViT-SAM model type. Options: l0, l1, l2, xl0, xl1")
    parser.add_argument("--image", type=str, default="dogs.jpg")
    parser.add_argument("--text_prompt", nargs='+', default=["a dog"])
    parser.add_argument("--out_path", type=str, default="yolo_efficientvit_sam_result.png")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    main(args)
