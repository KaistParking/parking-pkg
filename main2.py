import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

import time
import socket
from datetime import datetime
import argparse

import warnings
import sys
sys.path.append('./src')
warnings.filterwarnings(action='ignore')

try:
    from tag_recognition.watcher import Watcher, EKF
    from planning.planner import Planner
    from control.controller import Controller
    from communication.server import Client
except ImportError:
    from src.tag_recognition.watcher import Watcher, EKF
    from src.planning.planner import Planner
    from src.control.controller import Controller
    from src.communication.server import Client


watcher, planner, controller = None, None, None
client = None

ekf = None


def init_modules(map2planning_ratio=0.08):
    global watcher, planner, controller
    # draw map
    watcher = Watcher(img_size=(1920, 1080), tag_size=0.16)
    map_color = watcher.draw_map(color_full=(
        255, 255, 255), color_empty=(0, 0, 0))
    map_planning = cv2.resize(map_color, dsize=(
        0, 0), fx=map2planning_ratio, fy=map2planning_ratio)
    map_planning[np.where(map_planning != 255)] = 0
    # planning (m)
    planner = Planner(map_planning, meter_scale=0.01/map2planning_ratio)
    # control
    map_shape = map_color.shape
    controller = Controller(path=None, map_color=map_color,
                            map_size=(map_shape[1]/100, map_shape[0]/100))
    print("modules initialized")


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@static_vars(past_pose=None, past_time=None)
def estimate_vel(pose_in: np.ndarray):
    if estimate_vel.past_pose is None:
        estimate_vel.past_pose = pose_in.copy()
        estimate_vel.past_time = time.time()
        return 0.
    else:
        diff = pose_in - estimate_vel.past_pose
        diff_norm = np.linalg.norm(diff)
        dt = time.time() - estimate_vel.past_time
        estimate_vel.past_pose = pose_in.copy()
        estimate_vel.past_time = time.time()
        return diff_norm / dt if diff_norm * dt > 0 else 0


@static_vars(results_dir='results', img_idx=0)
def save_results(initial=False):
    if initial:
        now = datetime.now()
        time_now = '{}_{}_{}_{}_{}_{}'.\
            format(now.year, now.month, now.day,
                   now.hour, now.minute, now.second)
        save_results.results_dir = os.path.join(
            save_results.results_dir, time_now)
        os.mkdir(save_results.results_dir)
    else:
        plt.savefig(
            '{}/{}.jpg'.format(save_results.results_dir, save_results.img_idx))
        save_results.img_idx += 1


def find_parking_goal():
    global watcher, client
    empty_spots = {}
    while len(empty_spots) == 0:
        img_color, tags = client.receive_tags()
        _ = watcher.watch(img_color, tags=tags)
        empty_spots = watcher.find_empty_spots()
        cv2.imshow('tags', watcher.draw_tags())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.1)
    cv2.destroyAllWindows()
    for spot_id in [7, 5, 6, 7, 8, 9, 1, 2, 3, 4]:
        if spot_id in empty_spots:
            return empty_spots[spot_id]


@static_vars(im_idx=0)
def find_car_pose(car_id=0):
    global watcher, client, ekf
    while True:
        img_color, tags = client.receive_tags()

        # cv2.imwrite(
        #     'results/imgs/{}.jpg'.format(find_car_pose.im_idx), img_color)
        # find_car_pose.im_idx += 1

        tag_poses = watcher.watch(img_color, tags=tags)
        if tag_poses is None:
            continue
        elif car_id in tag_poses:
            trans, yaw = tag_poses[car_id]['trans'], tag_poses[car_id]['rot']
            pose = [trans[0], trans[1], yaw]
            # pose = ekf.apply(np.array(pose))
            return np.array(pose)


def planning_path(goal, car_id=0):
    global watcher, planner, client
    path = None
    while path is None:
        pose = find_car_pose(car_id=car_id)
        path = planner.plan_path(pose, goal)
    return path


def get_control(pose):
    global controller
    state = controller.update_pose(pose=pose, v=estimate_vel(pose[:2]))
    steer, accel = controller.estimate_control(state=state)
    return np.rad2deg(steer), accel


def jammed():
    global controller
    return len(controller.v) > 10 and np.mean(np.abs(controller.v[-10:])) < 0.1


def main(host='127.0.0.1', port=9999, modem='usbmodem', visualize=True, car_id=0, use_bt=False, save=True, connect=True):
    global watcher, planner, controller, client, ekf

    # 1) init modules
    init_modules(map2planning_ratio=0.10)
    print("modules initialized")

    # 2) start communication
    client = Client(host=host, port=port, use_bt=use_bt, basename=modem, vis=visualize, connect=connect)
    print("Host connected")

    # 3-1) find empty spots
    print('searching tags...')
    goal = find_parking_goal()
    print("empty spot found: {}".format(goal))
    if visualize:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.flip(watcher.draw_map(color_full=(
            100, 100, 100), color_empty=(0, 0, 0)), 0))
        plt.scatter([goal[0]*100], [watcher.map_h - goal[1]*100], s=50, c='c')
        plt.pause(1)
        plt.gca().clear()

    # EKF setup
    pose = None
    while pose is None:
        img_color, tags = client.receive_tags()
        tag_poses = watcher.watch(img_color, tags=tags)
        if tag_poses is None:
            continue
        elif car_id in tag_poses:
            trans, yaw = tag_poses[car_id]['trans'], tag_poses[car_id]['rot']
            pose = [trans[0], trans[1], yaw]
    ekf = EKF(initial_pose=pose,
              xy_obs_noise_std=1.5,
              initial_yaw_std=np.pi,
              forward_velocity_noise_std=1.5,
              yaw_rate_noise_std=0.17)

    # find path & control
    while True:
        # planning
        path = planning_path(goal)
        # reset controller
        controller.map_color = watcher.draw_map(
            color_full=(100, 100, 100), color_empty=(0, 0, 0))
        controller.init_path(path)

        while not controller.check_goal() and not jammed():
            # calculate control
            pose = find_car_pose(car_id=car_id)
            steer, accel = get_control(pose)

            # send msg
            handle = 'L' if steer > 0 else 'R'
            gear = 'F' if accel > 0 else 'B'
            msg = str('Q{}{}{:.2f},{:.2f}'.format(
                handle, gear, abs(steer), abs(accel)))
            client.send(msg)

            if visualize or save:
                plt.subplot(1, 2, 1)
                controller.show(ax=plt.gca())
                plt.subplot(1, 2, 2)
                ekf.show_results(ax=plt.gca())

                if save:
                    save_results()

                plt.pause(0.01)
                plt.subplot(1, 2, 1)
                plt.gca().clear()
                plt.subplot(1, 2, 2)
                plt.gca().clear()

        if controller.check_goal():
            break

    # parking ended
    msg = str('QLF0.00,0.00'.format(0.00, 0.00))
    client.send(msg)
    client.send(msg)
    client.send(msg)
    client.send(msg)
    client.send(msg)
    print('parking finished')
    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main (vehicle parking)')
    parser.add_argument('--hostname', type=str,
                        default=None, help='name of host')
    parser.add_argument('--modem', type=str,
                        default='usbmodem', help='name of host')
    parser.add_argument('--ip', type=str,
                        default='127.0.0.1', help='host ip adress')
    args = parser.parse_args()

    # HOST = '127.0.0.1'
    HOST = args.ip
    PORT = 9999
    if args.hostname is not None:
        HOST = socket.gethostbyname(args.hostname)

    main(host=HOST, port=PORT, modem=args.modem,
         visualize=True, car_id=0, use_bt=False, save=True, connect=True)
