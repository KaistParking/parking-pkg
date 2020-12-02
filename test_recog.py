import cv2
import matplotlib.pyplot as plt

import warnings
import sys
sys.path.append('./src')
warnings.filterwarnings(action='ignore')

try:
    from tag_recognition.watcher import Watcher
except ImportError:
    from src.tag_recognition.watcher import Watcher

# cap = cv2.VideoCapture('testing/parkinglot2.mov')
cap = cv2.VideoCapture('testing/test2.mov')
# cap = cv2.VideoCapture(0)

ret, img_color = cap.read()
while ret is False:
    ret, img_color = cap.read()
img_h, img_w = img_color.shape[:2]
watcher = Watcher(img_size=(1920, 1080), tag_size=0.16)

while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break

    ret, img_color = cap.read()
    if ret is None:
        break

    watcher.watch(img_color)

    img_tags = watcher.draw_tags()
    img_map = watcher.draw_map_with_tags()

    plt.imshow(img_map)
    if watcher.cam_trans is not None:
        print(watcher.cam_trans)
        plt.scatter([watcher.cam_trans[0]*100],
                    [watcher.cam_trans[1]*100], color='b')
    plt.pause(0.01)
    plt.cla()

    cv2.imshow('tags', img_tags)
    cv2.imshow('map', cv2.cvtColor(cv2.flip(img_map, 0), cv2.COLOR_BGR2RGB))

print("end")
