# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license


from test_utils import send_telegram_message
from ultralytics import YOLO

dataset_path = "D:\\UGM\\tugas akhir\\3. skripsi\\code\\datasets"
# dataset_path = '/home/ugm/Documents/zherk/datasets'


def on_epoch_end(trainer):
    ep = trainer.epoch + 1
    eps = trainer.epochs
    # if ep == 1 or ep % 50 == 0 or ep == eps:
    if ep % 50 == 0 or ep == eps:
        send_telegram_message(f"Epoch {ep} of {eps} done")


def main():
    # model = YOLO("yolo11n.yaml")
    model = YOLO("yolo11n-dist.yaml")
    # model = YOLO("yolo11n-obb.yaml")
    # model = YOLO("yolo11n-pose.yaml")
    # model = YOLO("yolo11n.pt")
    # model = YOLO("yolo11n-obb.pt")

    model.add_callback("on_train_epoch_end", on_epoch_end)

    # model.load(weights="../weights/yolo11n.pt")
    # model.load(weights="../weights/yolo11n_KITTI_pretrained_ep600.pt")
    # model.load(weights="../weights/yolov11n-distance_KITTI_ep300.pt")
    # model.load(weights="../runs/dist/train129/weights/best.pt")
    # model.load(weights="./best.pt")

    # results = model.train(data=dataset_path + "/coco8.yaml", epochs=5, imgsz=640)
    # results = model.train(data=dataset_path + "/coco8-dist.yaml", epochs=5, imgsz=640)
    # results = model.train(data=dataset_path + "/KITTI.yaml", epochs=5, imgsz=640, classes=[0,1,2,3,4,5,6])
    model.train(
        data=dataset_path + "/KITTI.yaml", epochs=300, imgsz=640, batch=32, workers=12, classes=[0, 1, 2, 3, 4, 5, 6]
    )
    # results = model.train(data=dataset_path + "/coco8-pose.yaml", epochs=5, imgsz=640)
    # print(results)

    # metrics = model.val(data=dataset_path + "/coco8.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6)
    # metrics = model.val(data=dataset_path + "/coco8-dist.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6)
    # metrics = model.val(data=dataset_path + "/KITTI.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6)
    # print(metrics)

    # detect_objects(model, "../datasets/street.jpg")
    # detect_objects(model, "../datasets/005992.png")
    # detect_objects(model, "../datasets/000007.png")
    # detect_objects(model, "../datasets/new-york.mp4")
    # detect_objects(model, "../datasets/kitti-track-video/0014.mp4", 10)
    # detect_objects(model, "../datasets/kitti-sequence2.mp4")

    # model.info(verbose=True, detailed=True)
    # summary(model.model, input_size=(1, 3, 640, 640))


if __name__ == "__main__":
    main()
