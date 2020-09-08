import cv2
import time
from PIL import Image
import numpy as np
from tools.deeplab import Deeplab
capture=cv2.VideoCapture("input_video/demo.mp4")
fps = capture.get(cv2.CAP_PROP_FPS)
size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("fps: {}\nsize: {}".format(fps, size))
# 读取视频时长（帧总数）
total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
print("[INFO] {} total frames in video".format(total))
# 调用VideoWrite（）函数
videoWrite = cv2.VideoWriter(filename='output/Video.avi', fourcc=cv2.VideoWriter_fourcc(*"MPEG"), fps=fps,frameSize= size,isColor=True,)
index = 0
deeplab = Deeplab()
while capture.isOpened():
    # 读取某一帧
    ref, frame = capture.read()
    frame=frame[:, :, ::-1]   
    # 进行检测
    frame = deeplab.main(frame,index,total)
    frame = frame[:, :, ::-1].astype(dtype=np.uint8)
    videoWrite.write(frame)
    cv2.show(frame)
    index += 1