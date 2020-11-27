import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np

from tag_recognition.watcher import Watcher3

# c920 params
fx = 646.47368
fy = 645.33399
cx = 298.07274
cy = 231.63218
param = [fx, fy, cx, cy]

# cap = cv2.VideoCapture('testing/parkinglot2.mov')
# cap = cv2.VideoCapture('testing/parkinglot3.mp4')
cap = cv2.VideoCapture(1)

_, img_color = cap.read()
img_h, img_w = img_color.shape[:2]
watcher = Watcher3(img_size=(img_w, img_h), tag_size=0.16)

while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break

    ret, img_color = cap.read()
    if ret is None:
        break

    tags = watcher.detector.detect(cv2.cvtColor(
        img_color, cv2.COLOR_BGR2GRAY), estimate_tag_pose=True, camera_params=param, tag_size=watcher.tag_size)

    vectors = []
    for tag in tags:
        r_cam = R.from_matrix(tag.pose_R)
        r_vec = r_cam.as_rotvec()
        # r_vec = np.array([r_vec[0], r_vec[2]])
        # r_vec = r_vec / np.linalg.norm(r_vec)
        vectors.append(r_vec)

    plt.xlim((-0.1, 0.1))
    plt.ylim((-0.1, 0.1))
    for vec in vectors:
        plt.scatter([vec[0], vec[1], 0], [0, 0, vec[2]], color='r')

    plt.pause(0.02)
    plt.cla()

    cv2.imshow('img', img_color)

    # img_map, camera_trans = watcher.watch_camera(img_color)
    # plt.imshow(img_map)
    # if camera_trans is not None:
    #     print(camera_trans)
    #     plt.scatter(camera_trans[:, 0]*100, camera_trans[:, 1]*100, color='r')
    # plt.pause(0.02)
    # plt.cla()
    # cv2.imshow('img', img_color)

    # watcher.watch(img_color)

    # img_tags = watcher.draw_tags()
    # img_map = watcher.draw_map_with_tags()

    # plt.imshow(img_map)
    # if watcher.cam_trans is not None:
    #     print(watcher.cam_trans)
    #     plt.scatter([watcher.cam_trans[0]*100],
    #                 [watcher.cam_trans[1]*100], color='r')
    # plt.pause(0.01)
    # plt.cla()

    # cv2.imshow('tags', img_tags)
    # cv2.imshow('map', img_map)

print("end")
