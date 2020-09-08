## 视频全量目标分析和建模
实现功能 ：视频目标检测、全景分割、目标跟踪，以及目标计数
### 解决思路
用Yolov4和Deepsort进行detection+track，使用Panoptic-DeepLab进行全景分割，检测器进行目标检测，然后跟踪器给每个不同目标一个id， 从而实现不同目标计数的效果。
### 分割
segmentation.py
所需环境：
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.3
- torchvision that matches the PyTorch installation.
You can install them together at pytorch.org to make sure of this.
- OpenCV, optional, needed by demo and visualization
### 检测及追踪
detection.py
所需环境:
- tensorflow-gpu==1.13.1
- keras==2.1.5
### 所需权重
链接：https://pan.baidu.com/s/17oxqc5YFZEb7KdjK-sOtPw 
提取码：kh72
yolov4.h5为yolov4的权重，须放在model_data文件夹下
另外两个为deeplab所需权重，须放在config文件夹下

Reference：
- https://github.com/bubbliiiing/yolov4-keras
- https://github.com/bowenc0221/panoptic-deeplab
