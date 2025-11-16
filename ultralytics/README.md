usage of yolov8-distance:

docs at https://docs.ultralytics.com/

![](https://github.com/billcao2000/yolov8-distance/blob/main/images/006631.png)

![](https://github.com/billcao2000/yolov8-distance/blob/main/images/006740.png)

![](https://github.com/billcao2000/yolov8-distance/blob/main/images/007343.png)

**Installation**

**to install this repo, please follow the instructions of the origional ultralytics**

**Dataset**

Using dataset from https://gitlab.com/EnginCZ/yolo-with-distance .

We provide two python files to transform this dataset into yolo format.

Config file at /data/kitti.yaml.

The structure of dataset is as follows:

Put the kitti folder under the root folder of this project.

--kitti

------images

------------test

------------train

------labels

------------test

------------train

To form seperate label files, use to_seperate_txt.py.

To build your dataset, download only the **train images** from kitti and make two copies in both train and test folders. Then use  delete_void.py to delete images without label files.



**terminal usage:**

**we offer pretrained weights in our releases page**

train:

```
yolo detect train data=kitti.yaml model=yolov8n.yaml batch=16 epochs=50
```

use pretrained yolov8n weights

```
yolo detect train model=yolov8n.pt data=kitti.yaml batch=16 epochs=100
```

predict:

```
yolo detect predict source="replace your source here" model=best.pt
```

for usage in jupyter notebook and custom settings, please refer to the ultralytics docs

**output:**

weights, plots at runs/detect/train

prediction at runs/detect/predict