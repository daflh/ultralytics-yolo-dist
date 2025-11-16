from ultralytics import YOLO
from torchinfo import summary
import cv2
from test_utils import detect_objects

dataset_path = 'D:\\UGM\\tugas akhir\\3. skripsi\\code\\datasets'
street_img = dataset_path + '\\street.jpg'
# street_img = dataset_path + '\\boats.jpg'

def main():
  # model = YOLO("yolo11n.yaml")
  model = YOLO("yolo11n-dist.yaml")
  # model = YOLO("yolo11n-obb.yaml")
  # model = YOLO("yolo11n-pose.yaml")
  # model = YOLO("yolo11n.pt")
  # model = YOLO("yolo11n-obb.pt")

  # model.load(weights="../weights/yolo11n.pt")
  # model.load(weights="../weights/yolo11n_KITTI_pretrained_ep600.pt")
  # model.load(weights="../runs/dist/train129/weights/best.pt")
  model.load(weights="./best.pt")
  
  # results = model.train(data=dataset_path + "/coco8.yaml", epochs=5, imgsz=640)
  # results = model.train(data=dataset_path + "/coco8-dist.yaml", epochs=5, imgsz=640)
  # results = model.train(data=dataset_path + "/KITTI.yaml", epochs=5, imgsz=640, workers=4, classes=[0,1,2,3,4,5,6])
  # results = model.train(data=dataset_path + "/dota8.yaml", epochs=5, imgsz=640)
  # results = model.train(data=dataset_path + "/coco8-pose.yaml", epochs=5, imgsz=640)
  # print(results)

  # metrics = model.val(data=dataset_path + "/coco8.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6)
  # metrics = model.val(data=dataset_path + "/coco8-dist.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6)
  # metrics = model.val(data=dataset_path + "/KITTI.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6)
  # print(metrics)
  
  # detect_objects(model, street_img)
  detect_objects(model, "../datasets/new-york.mp4")
  # detect_objects(model, "../datasets/kitti-sequence2.mp4")

  # model.info(verbose=True, detailed=True)
  # summary(model.model, input_size=(1, 3, 640, 640))

if __name__ == "__main__":
  main()
  