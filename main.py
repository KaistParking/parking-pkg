import serial
from bluetooth import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time

from tag_recognition.watcher import Watcher
from planning.planner import Planner
from control.controller import Controller
from communication.server import Client

import warnings
warnings.filterwarnings(action='ignore')


ser = serial.Serial(
    port='/dev/cu.usbmodem141201',
    baudrate=9600,
)


# client_socket = BluetoothSocket(RFCOMM)
# client_socket.connect(("98:d3:31:fd:3e:0d", 1))
# print("connected")

car_id = 0  # car tag_id
HOST = '127.0.0.1'
# HOST = '143.248.219.115'
# HOST = '192.168.1.101'
# HOST = '192.249.28.97'
PORT = 9999
use_bluetooth = False

results_dir = 'results'
results_idx = 0
while os.path.isdir(os.path.join(results_dir, str(results_idx))):
    results_idx += 1
results_dir = os.path.join(results_dir, str(results_idx))
os.mkdir(results_dir)


def init_modules():
    ratio = 0.08
    # draw map
    watcher_ = Watcher(img_size=(1920, 1080), tag_size=0.16)
    map_color = watcher_.draw_map(color_full=(
        255, 255, 255), color_empty=(0, 0, 0))
    map_planning = cv2.resize(map_color, dsize=(0, 0), fx=ratio, fy=ratio)
    map_planning[np.where(map_planning != 255)] = 0
    # planning (m)
    planner_ = Planner(map_planning, meter_scale=0.01/ratio)
    # control
    map_shape = map_color.shape
    car_ = Controller(path=None, map_color=map_color,
                      map_size=(map_shape[1]/100, map_shape[0]/100))
    return watcher_, planner_, car_


def vec2yaw(r):
    v = np.array(r[:2])
    v = v / np.linalg.norm(v)
    yaw = np.arccos(v[0]) if v[1] > 0 else -np.arccos(v[0])
    return yaw


past_pose = None
past_time = None


def estimate_vel(pose_in: np.ndarray):
    global past_pose, past_time
    if past_pose is None:
        past_pose = pose_in.copy()
        past_time = time.time()
        return 0.
    else:
        diff = pose_in - past_pose
        diff_norm = np.linalg.norm(diff)
        dt = time.time() - past_time
        past_pose = pose_in.copy()
        past_time = time.time()
        return diff_norm / dt if diff_norm * dt > 0 else 0


# init
watcher, planner, car = init_modules()
print("modules initialized")
plt.imshow(watcher.draw_map(color_full=(255, 255, 255), color_empty=(0, 0, 0)))
plt.show()

# start communication
client = Client(host=HOST, port=PORT, use_bluetooth=use_bluetooth)
print("Host connected")

# find empty spots
empty_spots = []
while len(empty_spots) == 0:
    img_color = client.receive()
    _ = watcher.watch(img_color)
    empty_spots = watcher.find_empty_spots()
    time.sleep(0.01)

goal = None
for spot_id in [9, 5, 6, 7, 8, 9, 1, 2, 3, 4]:
    if spot_id in empty_spots:
        goal = empty_spots[spot_id]
        break
# goal = empty_spots[6] if len(empty_spots) >= 7 else empty_spots[0]
print("empty spot found: {}".format(goal))

# find path & control
img_idx = 0
while True:
    path = None
    while path is None:
        img_color = client.receive()
        cv2.imshow('img', img_color)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        tag_poses = watcher.watch(img_color)
        if tag_poses is None or car_id not in tag_poses:
            continue
        trans, yaw = tag_poses[car_id]['trans'], tag_poses[car_id]['rot']
        pose = [trans[0], trans[1], yaw]
        path = planner.plan_path(pose, goal)

    if len(path.x_list) < 5:
        break

    car.map_color = watcher.draw_map(
        color_full=(100, 100, 100), color_empty=(0, 0, 0))
    car.init_path(path)
    while not car.check_goal() and not (len(car.v) > 10 and np.mean(np.abs(car.v[-10:])) < 0.1):
        img_color = client.receive()
        cv2.imshow('img', img_color)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        tag_poses = watcher.watch(img_color)
        if tag_poses is None or car_id not in tag_poses:
            continue
        trans, yaw = tag_poses[car_id]['trans'], tag_poses[car_id]['rot']
        pose = np.array([trans[0], trans[1], yaw])

        state = car.update_pose(pose=pose, v=estimate_vel(pose[:2]))
        steer, accel = car.estimate_control(state=state)
        if use_bluetooth:
            client.send_bluetooth_control(
                '{},{}'.format(str(steer), str(accel)))
            print('{},{}'.format(steer, accel))
        msg = str('{},{}'.format(str(steer), str(accel)))
        # client_socket.send(msg.encode(encoding="utf-8"))
        ser.write(msg.encode())

        car.show(ax=plt.gca())
        plt.savefig('{}/{}.jpg'.format(results_dir, img_idx))
        img_idx += 1
        plt.pause(0.02)
        plt.gca().clear()

if use_bluetooth:
    client.send_bluetooth_control('{},{}'.format(0.0, 0.0))
client.close()
print('parking finished')

img = cv2.imread('{}/{}.jpg'.format(results_dir, 1))
fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
out = cv2.VideoWriter('results/{}.avi'.format(results_idx),
                      fourcc, 10.0, (img.shape[1], img[0]))
for i in tqdm(range(img_idx), desc='video'):
    img = cv2.imread('{}/{}.jpg'.format(results_dir, i))
    cv2.imshow('img', img)
    cv2.waitKey(100)
    out.write(img)
out.release()
print('video saved')
