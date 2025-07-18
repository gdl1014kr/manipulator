import argparse
import time

import cv2
import numpy as np
import torch
import PIL.Image
from ultralytics import YOLO
from nanosam.utils.predictor import Predictor

def main(args):
    # 1. 모델 로딩
    print("--- Loading All Models ---")
    print(f"Loading YOLO-World model from: {args.yolo_model}")
    yolo_model = YOLO(args.yolo_model)
    yolo_model.to('cuda')
    
    print(f"Loading NanoSAM models: Encoder='{args.sam_encoder}', Decoder='{args.sam_decoder}'")
    sam_predictor = Predictor(args.sam_encoder, args.sam_decoder)
    print("All models loaded.")

    # 2. 워밍업
    print(f"\nWarming up the full pipeline with {args.warmup} iterations...")
    image_for_warmup = PIL.Image.open(args.image).convert("RGB")
    np_image_for_warmup = np.array(image_for_warmup)

    for _ in range(args.warmup):
        yolo_model.set_classes(args.text_prompt)
        results = yolo_model.predict(np_image_for_warmup, verbose=False)
        
        if len(results[0].boxes) > 0:
            sam_predictor.set_image(image_for_warmup)
            for box_obj in results[0].boxes:
                box = box_obj.xyxy[0].cpu().numpy()
                points = np.array([[box[0], box[1]], [box[2], box[3]]])
                point_labels = np.array([2, 3])
                sam_predictor.predict(points, point_labels)

    torch.cuda.synchronize()
    print("Warmup finished.")

    # 3. 전체 파이프라인 성능 측정
    print(f"\nRunning End-to-End pipeline measurement for {args.iters} iterations...")
    pipeline_latencies = []
    
    for _ in range(args.iters):
        torch.cuda.synchronize()
        start_time = time.time()
        
        image_pil = PIL.Image.open(args.image).convert("RGB")
        image_np = np.array(image_pil)

        yolo_model.set_classes(args.text_prompt)
        results = yolo_model.predict(image_np, verbose=False)

        if len(results[0].boxes) == 0:
            print(f"Warning: No objects detected for prompts '{args.text_prompt}'. Skipping.")
            continue
        
        sam_predictor.set_image(image_pil)
        overlay_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        for box_obj in results[0].boxes:
            box = box_obj.xyxy[0].cpu().numpy()
            
            # NanoSAM 추론
            points = np.array([[box[0], box[1]], [box[2], box[3]]])
            point_labels = np.array([2, 3])
            mask, _, _ = sam_predictor.predict(points, point_labels)
            
            mask_processed = (mask[0, 0] > 0).cpu().numpy()
            green_mask = np.zeros_like(overlay_img, dtype=np.uint8)
            green_mask[mask_processed] = [0, 255, 0]
            alpha = 0.5
            overlay_img = cv2.addWeighted(overlay_img, 1, green_mask, alpha, 0)

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # --- ✨ 새로운 기능: 바운딩 박스에 프롬프트 텍스트 추가 ✨ ---
            conf = box_obj.conf[0].cpu().item()  # 신뢰도 점수
            cls_id = int(box_obj.cls[0].cpu().item())  # 프롬프트 인덱스
            prompt_text = args.text_prompt[cls_id]  # 인덱스로 프롬프트 조회
            
            label = f"{prompt_text}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            text_color = (255, 255, 255)
            bg_color = (0, 0, 255) # 박스와 동일한 빨간색 배경
            
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, 2)
            cv2.rectangle(overlay_img, (x1, y1 - text_height - 10), (x1 + text_width, y1), bg_color, -1)
            cv2.putText(overlay_img, label, (x1, y1 - 5), font, font_scale, text_color, 2)
            # ----------------------------------------------------

        cv2.imwrite(args.out_path, overlay_img)

        torch.cuda.synchronize()
        end_time = time.time()
        pipeline_latencies.append((end_time - start_time) * 1000)

    if not pipeline_latencies:
        print("Could not measure performance because no objects were detected.")
        return

    # ... 나머지 계산, 출력, 시각화 코드는 동일 ...
    avg_pipeline_latency_ms = np.mean(pipeline_latencies)
    std_pipeline_latency_ms = np.std(pipeline_latencies)
    pipeline_fps = 1000.0 / avg_pipeline_latency_ms if avg_pipeline_latency_ms > 0 else 0

    final_img = cv2.imread(args.out_path)
    perf_text_latency = f"Avg Pipeline Latency: {avg_pipeline_latency_ms:.2f} ms"
    perf_text_fps = f"Pipeline FPS: {pipeline_fps:.2f}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)

    cv2.rectangle(final_img, (5, 5), (500, 80), bg_color, -1)
    cv2.putText(final_img, perf_text_latency, (10, 40), font, font_scale, text_color, thickness)
    cv2.putText(final_img, perf_text_fps, (10, 70), font, font_scale, text_color, thickness)
    
    cv2.imwrite(args.out_path, final_img)
    
    print("\n--- Performance Metrics (YOLO-World + NanoSAM Pipeline) ---")
    print(f"Number of Iterations: {args.iters}")
    print(f"Average End-to-End Pipeline Latency: {avg_pipeline_latency_ms:.2f} ms (+/- {std_pipeline_latency_ms:.2f} ms)")
    print(f"End-to-End Pipeline FPS: {pipeline_fps:.2f}")
    print(f"\nResult saved in {args.out_path}")
    
    cv2.imshow("YOLO-World + NanoSAM Pipeline", final_img)
    print("Displaying result. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO-World + NanoSAM End-to-End Pipeline Script")
    parser.add_argument("--yolo_model", type=str, default="yolov8s-worldv2.pt", help="Path to the YOLO-World model file.")
    parser.add_argument("--sam_encoder", type=str, default="../nvidia/nanosam/data/image_encoder_fp16.engine", help="Path to the NanoSAM encoder engine.")
    parser.add_argument("--sam_decoder", type=str, default="../nvidia/nanosam/data/mask_decoder_fp16.engine", help="Path to the NanoSAM decoder engine.")
    parser.add_argument("--image", type=str, default="dogs.jpg", help="Path to the input image.")
    # 여러 객체 동시 탐지를 위해 리스트로 받되, 가장 대표적인 프롬프트 하나를 default로 설정
    parser.add_argument("--text_prompt", nargs='+', default=["a dog"], help="List of text prompts for object detection.")
    parser.add_argument("--out_path", type=str, default="yolo_sam_pipeline_result.png", help="Path to save the output image.")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations.")
    parser.add_argument("--iters", type=int, default=20, help="Number of measurement iterations.")
    args = parser.parse_args()
    main(args)
