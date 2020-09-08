import cv2
import time
from tools.deeplab import Deeplab
capture=cv2.VideoCapture("input/软件杯决赛demo.mp4")
fps = int(capture.get(cv2.CAP_PROP_FPS))
size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("fps: {}\nsize: {}".format(fps, size))
# 读取视频时长（帧总数）
total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
print("[INFO] {} total frames in video".format(total))
# 调用VideoWrite（）函数
videoWrite = cv2.VideoWriter('output/outputVideo.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
index = 0
deeplab = Deeplab()
while capture.isOpened():
    # 读取某一帧
    ref, frame = capture.read()
    # 进行检测
    frame = deeplab.main(frame,index,total)
    videoWrite.write(frame)
    cv2.imshow("video", frame)
    index += 1

capture.release()