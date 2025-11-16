import torch
import cv2
import time
import os
from ultralytics import YOLO
from torchinfo import summary
import sys

def detect_objects(model, input_path):
    # model.info()
    # print(model)

    # summary(model.model, input_size=(1, 3, 640, 640))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"Using device: {device}")

    # --- Determine file type ---
    ext = os.path.splitext(input_path)[1].lower()
    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv']

    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return

    # --- VIDEO MODE ---
    if is_video:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Error: Unable to open video file.")
            return

        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Start time for FPS calculation
            frame_start = time.time()

            # Perform detection
            results = model(frame)

            # Annotate results
            annotated_frame = results[0].plot()
            print(results[0].distances)

            # Compute FPS
            frame_time = time.time() - frame_start
            fps = 1.0 / frame_time if frame_time > 0 else 0

            # Show FPS on the frame
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("YOLO Detection (Video)", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

            # time.sleep(5)

        cap.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames in {time.time() - start_time:.2f}s")

    # --- IMAGE MODE ---
    else:
        img = cv2.imread(input_path)
        if img is None:
            print("Error: Unable to read image.")
            return

        results = model(img)
        annotated_img = results[0].plot()

        # Show result
        cv2.imshow("YOLO Detection (Image)", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Optionally save
        # save_path = "output_detected.jpg"
        # cv2.imwrite(save_path, annotated_img)
        # print(f"Saved detection result to {save_path}")


# arg1 = sys.argv[1] if len(sys.argv) > 1 else None
# arg2 = sys.argv[2] if len(sys.argv) > 2 else None

# if __name__ == "__main__":
#     weights_path = arg1 if arg1 else "./best.pt"
#     # input_path = "datasets/000007.png"
#     # input_path = "datasets/new-york.mp4"
#     input_path = arg2 if arg2 else "../datasets/new-york.mp4"
#     # input_path = arg2 if arg2 else "../datasets/kitti-sequence2.mp4"
    
#     model = YOLO(weights_path, verbose=True)
#     detect_objects(model, input_path)
    