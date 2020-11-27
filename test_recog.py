import cv2
import matplotlib.pyplot as plt
from tag_recognition.watcher import Watcher3

# cap = cv2.VideoCapture('testing/parkinglot2.mov')
# cap = cv2.VideoCapture('testing/parkinglot3.mp4')
cap = cv2.VideoCapture(1)

_, img_color = cap.read()
img_h, img_w = img_color.shape[:2]
watcher = Watcher3(img_size=(640, 480), tag_size=0.16)

while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break

    ret, img_color = cap.read()
    if ret is None:
        break

    # img_map, camera_trans = watcher.watch_camera(img_color)
    # plt.imshow(img_map)
    # if camera_trans is not None:
    #     print(camera_trans)
    #     plt.scatter(camera_trans[:, 0]*100, camera_trans[:, 1]*100, color='r')
    # plt.pause(0.02)
    # plt.cla()
    # cv2.imshow('img', img_color)

    img_color = cv2.resize(img_color, (640, 480))

    watcher.watch(img_color)

    img_tags = watcher.draw_tags()
    img_map = watcher.draw_map_with_tags()

    plt.imshow(img_map)
    if watcher.cam_trans is not None:
        # if len(watcher.tags) != 0:
        #     print(watcher.tags[0].pose_t[:, 0])
        print(watcher.cam_trans)
        plt.scatter([watcher.cam_trans[0]*100],
                    [watcher.cam_trans[1]*100], color='r')
    plt.pause(0.01)
    plt.cla()

    cv2.imshow('tags', img_tags)
    # cv2.imshow('map', img_map)

print("end")
