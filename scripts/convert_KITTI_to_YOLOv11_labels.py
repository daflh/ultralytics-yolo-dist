import os
from tqdm import tqdm

KITTI_IMAGES_DIR = "D:\\UGM\\tugas akhir\\datasets\\KITTI2017\\images\\train"
KITTI_LABELS_DIR = "D:\\UGM\\tugas akhir\\datasets\\KITTI2017\\labels_KITTI\\train"
YOLO_LABELS_OUTPUT_DIR = "D:\\UGM\\tugas akhir\\datasets\\KITTI2017\\labels_YOLO\\train"
KITTI_CLASSES = [ # order matters
    "Pedestrian", "Car", "Van", "Truck", "Person_sitting",
    "Cyclist", "Tram", "Misc", "DontCare"
]
ADD_DISTANCE_VALUES = True  # if False, output normal YOLO format
INCLUDE_DONTCARE = False
INCLUDE_MISC = False


def load_image_size(image_path):
    from PIL import Image
    with Image.open(image_path) as img:
        return img.width, img.height


def kitti_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height


def convert_kitti_label(kitti_line, img_w, img_h):
    parts = kitti_line.strip().split()
    cls_name = parts[0]

    if cls_name == "DontCare" and not INCLUDE_DONTCARE:
        return None
    if cls_name == "Misc" and not INCLUDE_MISC:
        return None

    if cls_name not in KITTI_CLASSES:
        print(f"Warning: Class '{cls_name}' not in KITTI_CLASSES, skipping")
        return None

    cls_id = KITTI_CLASSES.index(cls_name)

    xmin, ymin, xmax, ymax = map(float, parts[4:8])
    x, y, w, h = kitti_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h)

    if ADD_DISTANCE_VALUES:
        X, Y, Z = map(float, parts[11:14])
        return f"{cls_id} {x} {y} {w} {h} {X} {Y} {Z}"
    else:
        return f"{cls_id} {x} {y} {w} {h}"


def convert_all():
    os.makedirs(YOLO_LABELS_OUTPUT_DIR, exist_ok=True)
    label_files = sorted([f for f in os.listdir(KITTI_LABELS_DIR) if f.endswith(".txt")])

    for filename in tqdm(label_files, desc="Converting labels KITTI to YOLO"):
        kitti_label_path = os.path.join(KITTI_LABELS_DIR, filename)
        image_path = os.path.join(KITTI_IMAGES_DIR, filename.replace(".txt", ".png"))

        # some KITTI sets use .jpg instead of .png
        if not os.path.exists(image_path):
            image_path = image_path.replace(".png", ".jpg")
        if not os.path.exists(image_path):
            print(f"Warning: Image not found for {filename}, skipping")
            continue

        img_w, img_h = load_image_size(image_path)

        with open(kitti_label_path, "r") as f:
            lines = f.readlines()

        output_lines = []
        for line in lines:
            converted = convert_kitti_label(line, img_w, img_h)
            if converted:
                output_lines.append(converted)

        # write YOLO label
        output_path = os.path.join(YOLO_LABELS_OUTPUT_DIR, filename)
        with open(output_path, "w") as out:
            out.write("\n".join(output_lines))


if __name__ == "__main__":
    convert_all()
    print("Conversion complete!")
