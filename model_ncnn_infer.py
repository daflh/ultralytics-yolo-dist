#!/usr/bin/env python3
from ultralytics import YOLO
import torch
import cv2
import time
from test_utils import BirdEyeView

MODEL_PATH = "yolo11n-dist_ncnn_model"
VIDEO_SOURCE = 1  # use device id or path to video file
TARGET_FPS = 10  # set maximum FPS
SHOW_BEV = True  # bird's eye view

def main():
    model = YOLO(MODEL_PATH)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if SHOW_BEV:
        bev_width = 500
        bev_height = 500
        
        bev = BirdEyeView()

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Error: Unable to open capture device")
        return

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.time()

        results = model.predict(frame, device=device)
        annotated_frame = results[0].plot(conf=False)

        infer_time = time.time() - frame_start
        sleep_time = (1.0 / TARGET_FPS) - infer_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        frame_time = time.time() - frame_start
        fps = 1.0 / frame_time if frame_time > 0 else 0

        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("YOLO Detection Result", annotated_frame)

        if SHOW_BEV:
            res = results[0]

            bev.reset()
            for i in range(len(res.boxes.xyxy)):
                bev.draw_box({
                    "object_id": res.boxes.cls[i],
                    "bbox": res.boxes.xyxy[i],
                    "distance": res.distances[i],
                    "img_size": res.orig_shape[::-1]
                })

            bev_image = bev.get_image()
            bev_image = cv2.resize(bev_image, (bev_width, bev_height))

            cv2.imshow("YOLO Bird's Eye View", bev_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames in {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
    