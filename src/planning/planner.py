import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

try:
    import planning.hybrid_a_star as a_star
except:
    import src.planning.hybrid_a_star as a_star


def set_planner_params(scale):
    a_star.XY_GRID_RESOLUTION /= scale
    a_star.MOTION_RESOLUTION /= scale
    a_star.VR /= scale


class Planner:
    def __init__(self, map_color=None, meter_scale=0.1):
        self.meter_scale = meter_scale
        self.map_color = map_color
        self.img_scale = None
        self.map_gray = None
        self.map_show = None
        self.path = None
        self.start = None
        self.end = None
        self.map_obs_x = None
        self.map_obs_y = None

        map_shape = map_color.shape
        self.map_w = map_shape[1] * self.meter_scale
        self.map_h = map_shape[0] * self.meter_scale
        self.map_obs_x, self.map_obs_y = self.extract_map(self.map_color)

        set_planner_params(self.meter_scale)

    def extract_map(self, map_img):
        if map_img.shape[-1] == 3:
            self.map_gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
        else:
            self.map_gray = map_img.copy()
        objects_pixel = np.where(self.map_gray != 255)
        self.map_obs_x = objects_pixel[1].tolist()
        self.map_obs_y = objects_pixel[0].tolist()
        return self.map_obs_x, self.map_obs_y

    def plan_path(self, start, end):
        self.start = start.copy()
        self.end = end.copy()
        start_ = start.copy()
        end_ = end.copy()
        start_[0] = start[0] / self.meter_scale
        start_[1] = start[1] / self.meter_scale
        end_[0] = end[0] / self.meter_scale
        end_[1] = end[1] / self.meter_scale
        self.path = a_star.hybrid_a_star_planning(start_, end_, self.map_obs_x, self.map_obs_y)
        if self.path == ([], [], []):
            return None
        self.path.x_list = np.asarray(self.path.x_list) * self.meter_scale
        self.path.y_list = np.asarray(self.path.y_list) * self.meter_scale
        self.path.yaw_list = np.asarray(self.path.yaw_list)
        return self.path

    def get_path(self):
        return self.path

    def show_path(self, ax=plt.gca()):
        ax.imshow(cv2.flip(self.map_color, 0), extent=[0, self.map_w, 0, self.map_h])
        ax.plot(self.path.x_list, self.path.y_list)
        length = 0.5
        ax.arrow(self.start[0], self.start[1], length * math.cos(self.start[2]), length * math.sin(self.start[2]),
                 label='start', color='r', alpha=0.7, linewidth=5, head_width=.2, head_length=.2)
        ax.arrow(self.end[0], self.end[1], length * math.cos(self.end[2]), length * math.sin(self.end[2]),
                 label='goal', color='b', alpha=0.7, linewidth=5, head_width=.2, head_length=.2)
        ax.legend(loc='upper right')
        plt.title('planned path')
