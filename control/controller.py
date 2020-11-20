import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import math

import control.model_predictive_speed_and_steer_control as mpc
import copy


class Controller:
    def __init__(self, path, map_color, map_size):
        self.map_color = map_color.copy()
        self.map_w = map_size[0]
        self.map_h = map_size[1]
        self.dl = 1.0  # course tick

        self.times = []  # time
        self.plan_x = None
        self.plan_y = None
        self.plan_yaw = None
        self.speed_profile = None
        self.initial_state = None
        self.goal = None
        self.x = []  # pose_x
        self.y = []  # pose_y
        self.yaw = []  # pose_yaw
        self.v = []  # velocity
        self.d = []  # steer
        self.a = []  # accel
        self.state = None
        self.target_ind = None
        self.odelta = None
        self.oa = None
        self.init_path(copy.deepcopy(path))

    def init_path(self, path):
        self.plan_x = path.x_list
        self.plan_y = path.y_list
        self.plan_yaw = path.yaw_list
        self.speed_profile = mpc.calc_speed_profile(
            self.plan_x, self.plan_y, self.plan_yaw, mpc.TARGET_SPEED)
        self.initial_state = mpc.State(
            x=self.plan_x[0], y=self.plan_y[0], yaw=self.plan_yaw[0], v=0.0)
        self.goal = [self.plan_x[-1], self.plan_y[-1], self.plan_yaw[-1]]

        self.x = []  # pose_x
        self.y = []  # pose_y
        self.yaw = []  # pose_yaw
        self.v = []  # velocity
        self.d = []  # steer
        self.a = []  # accel
        self.state = copy.deepcopy(self.initial_state)
        if self.state.yaw - self.plan_yaw[0] >= math.pi:
            self.state.yaw -= math.pi * 2.0
        elif self.state.yaw - self.plan_yaw[0] <= -math.pi:
            self.state.yaw += math.pi * 2.0
        self.x.append(self.state.x)
        self.y.append(self.state.y)
        self.yaw.append(self.state.yaw)
        self.v.append(self.state.v)
        self.times.append(time.time())
        self.d.append(0.0)
        self.a.append(0.0)

        self.target_ind, _ = mpc.calc_nearest_index(
            self.state, self.plan_x, self.plan_y, self.plan_yaw, 0)
        self.odelta = None
        self.oa = None
        self.plan_yaw = mpc.smooth_yaw(self.plan_yaw)

    def update_pose(self, pose=None, v=None):
        if pose is None:
            self.state = mpc.update_state(self.state, self.a[-1], self.d[-1], time.time() - self.times[-1])
        else:
            self.state.x = pose[0]
            self.state.y = pose[1]
            self.state.yaw = pose[2]
            self.state.v = v if v is not None else self.state.v
        return self.state

    def estimate_control(self, state=None):
        if state is not None:
            self.state = state
        mpc.DT = time.time() - self.times[-1]
        # calculate control output
        xref, self.target_ind, dref = mpc.calc_ref_trajectory(
            self.state, self.plan_x, self.plan_y, self.plan_yaw, [], self.speed_profile, self.dl, self.target_ind)
        x0 = [self.state.x, self.state.y, self.state.v, self.state.yaw]  # current state
        self.oa, self.odelta, ox, oy, oyaw, ov = \
            mpc.iterative_linear_mpc_control(xref, x0, dref, self.oa, self.odelta)

        # save states
        di, ai = self.odelta[0], self.oa[0]

        self.x.append(self.state.x)
        self.y.append(self.state.y)
        self.yaw.append(self.state.yaw)
        self.v.append(self.state.v)
        self.times.append(time.time())
        self.d.append(di)
        self.a.append(ai)

        return di, ai

    def check_goal(self):
        return mpc.check_goal(self.state, self.goal, self.target_ind, len(self.plan_x))

    def show(self, ax=plt.gca()):
        plt.title("Time[s]: {}s, speed[km/h]: {}m/s".format(
            round(self.times[-1] - self.times[0], 2), round(self.v[-1], 2)))

        ax.imshow(cv2.flip(self.map_color, 0), extent=[0, self.map_w, 0, self.map_h])

        length = 0.5
        ax.plot(self.plan_x, self.plan_y, "-r", label="course")
        ax.plot(self.x, self.y, "ob", label="trajectory")
        ax.plot(self.plan_x[self.target_ind], self.plan_y[self.target_ind], "xg", label="target")
        ax.arrow(self.initial_state.x, self.initial_state.y,
                 length * math.cos(self.initial_state.yaw), length * math.sin(self.initial_state.yaw),
                 label='start', color='r', alpha=0.7, linewidth=5, head_width=.2, head_length=.2)
        ax.arrow(self.goal[0], self.goal[1], length * math.cos(self.goal[2]), length * math.sin(self.goal[2]),
                 label='goal', color='b', alpha=0.7, linewidth=5, head_width=.2, head_length=.2)

        mpc.plot_car(self.state.x, self.state.y, self.state.yaw, steer=self.d[-1], ax=ax)

        ax.text(0.1, self.map_h-0.3, 'Steer: {} degree, Accel: {}m/s^2'.format(
            round(np.rad2deg(self.d[-1]), 2), round(self.a[-1], 2)), color='r')

        # ax.axis("equal")
        # ax.grid(False)
