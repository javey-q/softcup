import matplotlib.pyplot as plt
from yolo import YOLO
import cv2
import time
from PIL import Image
import numpy as np
import  os

# 调用摄像头
capture=cv2.VideoCapture("input_video/软件杯决赛demo.mp4")
capture1 = cv2.VideoCapture("output/Video.avi")
fps = int(capture1.get(cv2.CAP_PROP_FPS))
size = (int(capture1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("fps: {}\nsize: {}".format(fps, size))

# 读取视频时长（帧总数）
total = int(capture1.get(cv2.CAP_PROP_FRAME_COUNT))
print("[INFO] {} total frames in video".format(total))
# 调用VideoWrite（）函数
videoWrite = cv2.VideoWriter('output/outputVideo.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, size)

diclass = {'car': 0,'bus': 0, 'person': 0, 'bike': 0, 'truck': 0, 'motor': 0,'train': 0,'rider':0,'traffic sign': 0, 'traffic light': 0}
txt = "output/totalCount.txt"  # 将要输出保存的文件地址

yolo = YOLO()
index = 0
while capture1.isOpened():
    t1 = time.time()
    # 读取某一帧
    ref, frame = capture.read()
    _, frame1 = capture1.read()
    # 进行检测
    frame,class_dict = yolo.detect_image(frame,frame1)
    with open(txt,"w") as f:
        for k in class_dict.keys():
            f.write(k + ': ' + str(class_dict[k]))  # 将字符串写入文件中
            f.write("\n")  # 换行
            
    fps = (fps + (1. / (time.time() - t1))) / 2
    #print("fps= %.2f" % (fps))
    frame = cv2.putText(frame, "fps= %.2f" % (fps), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (144,238,144), 2)
    
    videoWrite.write(frame)
    cv2.show(frame)
    print(index)
    index += 1
capture.release()
capture1.release()
yolo.close_session()