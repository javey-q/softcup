#-------------------------------------#
#       调用摄像头检测
#-------------------------------------#
from yolo import YOLO
import cv2
import time
yolo = YOLO()
import  os
# 调用摄像头
#capture=cv2.VideoCapture(0)
capture=cv2.VideoCapture("input_video/softcup.mp4")

fps = int(capture.get(cv2.CAP_PROP_FPS))
size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("fps: {}\nsize: {}".format(fps, size))

# 读取视频时长（帧总数）
total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
print("[INFO] {} total frames in video".format(total))
# 调用VideoWrite（）函数
videoWrite = cv2.VideoWriter('output/outputVideo.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

diclass = {'car': 0,'bus': 0, 'person': 0, 'bike': 0, 'truck': 0, 'motor': 0,'train': 0,'rider':0,'traffic sign': 0, 'traffic light': 0}
txt = "output/totalCount.txt"  # 将要输出保存的文件地址

if not os.path.exists('output/imageSeg'):
        os.makedirs('output/imageSeg')
        
for dir in diclass.keys():
    path_dir = 'output/imageSeg/' + dir
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

while (True):
    # 读取某一帧
    ref, frame = capture.read()

    # 进行检测
    frame,class_dict = yolo.detect_image(frame)

    with open(txt,"w") as f:
        for k in class_dict.keys():
            f.write(k + ': ' + str(class_dict[k]))  # 将字符串写入文件中
            f.write("\n")  # 换行

    videoWrite.write(frame)
    
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,10))
    plt.imshow(frame)
    plt.axis('on')
    plt.show()

    c = cv2.waitKey(30) & 0xff
    if c == 27:
        capture.release()
        break


yolo.close_session()