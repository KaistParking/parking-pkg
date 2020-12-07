import numpy as np
import cv2
from scipy.spatial.transform import Rotation

from pupil_apriltags import Detector
import time
import matplotlib.pyplot as plt

try:
    from tag_recognition.kalman_filters import ExtendedKalmanFilter
    from tag_recognition.utils import normalize_angles
except ImportError:
    from src.tag_recognition.kalman_filters import ExtendedKalmanFilter
    from src.tag_recognition.utils import normalize_angles

# c920 params - old
# fx = 646.47368
# fy = 645.33399
# cx = 298.07274 * 1920/640
# cy = 231.63218 * 1080/480
# param = [fx, fy, cx, cy]
# param = [686.42860, 687.41792, 740.56855, 624.89619]
# param = [686.42860, 687.41792, 740.56855, 624.89619]

# c920
param = [1444.30087, 1447.86206, 959.50000, 539.50000]
# param = [686.42860, 687.41792, 959.50000, 539.50000]

# drone
# param = [1090.17316, 1090.67229, 969.14173, 525.16701]

parking_spots = {
    1: [[268, 152], [267, 0], [358, 0], [356, 153]],
    2: [[180, 152], [179, 0], [267, 0], [268, 152]],
    3: [[91, 151], [91, 0], [179, 0], [180, 152]],
    4: [[0, 150], [0, 0], [91, 0], [91, 151]],
    5: [[360, 408], [361, 544], [454, 543], [451, 410]],
    6: [[271, 409], [274, 543], [361, 544], [360, 408]],
    7: [[180, 408], [182, 543], [274, 543], [271, 409]],
    8: [[92, 405], [92, 542], [182, 543], [180, 408]],
    9: [[0, 403], [0, 543], [92, 542], [92, 405]]
}

tag_rot_base = np.array([0., 0., 0.])  # rad
tag_status = {
    1:  {'id': 1,   'trans': np.array([3.091, 0.80, 0]),    'rot': tag_rot_base.copy()},
    2:  {'id': 2,   'trans': np.array([2.212, 0.734, 0]),   'rot': tag_rot_base.copy()},
    3:  {'id': 3,   'trans': np.array([1.355, 0.78, 0]),    'rot': tag_rot_base.copy()},
    4:  {'id': 4,   'trans': np.array([0.432, 0.838, 0]),   'rot': tag_rot_base.copy()},
    5:  {'id': 5,   'trans': np.array([4.077, 4.90, 0]),    'rot': tag_rot_base.copy()},
    6:  {'id': 6,   'trans': np.array([3.172, 4.89, 0]),    'rot': tag_rot_base.copy()},
    7:  {'id': 7,   'trans': np.array([2.269, 4.86, 0]),    'rot': tag_rot_base.copy()},
    8:  {'id': 8,   'trans': np.array([1.368, 4.88, 0]),    'rot': tag_rot_base.copy()},
    9:  {'id': 9,   'trans': np.array([0.495, 4.81, 0]),    'rot': tag_rot_base.copy()},
    10: {'id': 10,  'trans': np.array([2.68, 1.70, 0]),     'rot': tag_rot_base.copy()},
    11: {'id': 11,  'trans': np.array([1.797, 1.70, 0]),    'rot': tag_rot_base.copy()},
    12: {'id': 12,  'trans': np.array([0.905, 1.69, 0]),    'rot': tag_rot_base.copy()},
    13: {'id': 13,  'trans': np.array([0.923, 3.87, 0]),    'rot': tag_rot_base.copy()},
    14: {'id': 14,  'trans': np.array([1.802, 3.89, 0]),    'rot': tag_rot_base.copy()},
    15: {'id': 15,  'trans': np.array([2.713, 3.90, 0]),    'rot': tag_rot_base.copy()},
    16: {'id': 16,  'trans': np.array([3.602, 3.89, 0]),    'rot': tag_rot_base.copy()},
}


class Watcher:
    def __init__(self, img_size=(1920, 1080), tag_size=0.16):
        self.map_w = np.max([np.asarray(val)[:, 0]
                             for val in parking_spots.values()])
        self.map_h = np.max([np.asarray(val)[:, 1]
                             for val in parking_spots.values()])

        self.detector = Detector(
            families='tag36h11', nthreads=4, quad_decimate=1.0)

        self.img_size = img_size
        self.tag_size = tag_size

        self.cam_trans = None
        self.cam_rot = None

        self.img = None
        self.tags = None
        self.tag_poses = None
        self.tag_history = {}

        self.trans_stack = []
        self.rot_stack = []

        # self.ekf = EKF(initial_pose=np.array([0., 0., 0.]))

    def find_camera_frame(self):
        cam_translation = []
        cam_rotation = []

        for tag in self.tags:
            if tag.tag_id not in tag_status:
                continue

            # camera translation
            pose_world2cam = tag_status[tag.tag_id]['trans'] - \
                np.matmul(np.linalg.inv(tag.pose_R),
                          tag.pose_t[:, 0]) * np.array([1., -1., -1.])
            cam_translation.append(pose_world2cam)

            # camera rotation (base: camera coordinate)
            cam_rotation.append(np.linalg.inv(tag.pose_R))

        if len(cam_translation) == 0:
            return None, None

        trans_mean = np.mean(cam_translation, axis=0)
        rot_mean = Rotation.from_matrix(cam_rotation).mean().as_matrix()
        return trans_mean, rot_mean

    def calculate_tag_pose(self):
        results = {}
        for tag in self.tags:
            # tag translation
            pose_world2tag = self.cam_trans + \
                np.matmul(self.cam_rot,
                          tag.pose_t[:, 0]) * np.array([1., -1., -1.])
            # tag rotation(yaw)
            # rot_world2tag = Rotation.from_matrix(np.matmul(self.cam_rot, tag.pose_R))
            # tag_yaw = rot_world2tag.as_rotvec()[2]
            tag_yaw = \
                Rotation.from_matrix(self.cam_rot).as_rotvec()[
                    2] + Rotation.from_matrix(tag.pose_R).as_rotvec()[2]

            results[tag.tag_id] = {
                'trans': pose_world2tag, 'rot': -tag_yaw + np.pi/2}
            self.tag_history[tag.tag_id] = {
                'trans': pose_world2tag, 'rot': -tag_yaw + np.pi/2}
        return results

    def watch(self, img_color, tags=None):
        # detect tags
        if tags is None:
            self.img = img_color.copy()
            img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.resize(img_gray, self.img_size)
            self.tags = self.detector.detect(
                img_gray, estimate_tag_pose=True, camera_params=param, tag_size=self.tag_size)
        else:
            self.img = img_color.copy()
            self.tags = tags

        # find camera pose
        cam_trans, cam_rot = self.find_camera_frame()
        if cam_trans is None:
            return None
        else:
            self.trans_stack.append(cam_trans)
            self.rot_stack.append(cam_rot)
            if len(self.trans_stack) >= 3:
                self.trans_stack = self.trans_stack[-3:]
            if len(self.rot_stack) >= 3:
                self.rot_stack = self.rot_stack[-3:]

            self.cam_trans = np.mean(np.asarray(self.trans_stack), axis=0)
            self.cam_rot = Rotation.from_matrix(
                self.rot_stack).mean().as_matrix()

            # self.cam_trans = self.ekf.apply(self.cam_trans)

            self.tag_poses = self.calculate_tag_pose()
            return self.tag_poses

    def draw_tags(self):
        img_show = self.img.copy()
        for tag in self.tags:
            corners = tag.corners
            img_show = cv2.polylines(
                img_show, [corners.astype(np.int64)], True, (0, 0, 255), 10)
            cv2.putText(img_show, str(tag.tag_id), (int(corners[0][0]), int(corners[0][1])),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0), 4)
        return img_show

    def draw_map(self, color_full=(100, 100, 100), color_empty=(240, 240, 0)):
        img_show = 255 * \
            np.ones(shape=(self.map_h, self.map_w, 3), dtype=np.uint8)
        for spot_id, corners in parking_spots.items():
            corners = np.asarray(corners, np.int64)
            if spot_id not in self.tag_history:
                img_show = cv2.fillPoly(img_show, [corners], color_full)
                img_show = cv2.polylines(
                    img_show, [corners], False, color_empty, 20)
                # img_show = cv2.line(
                #     img_show, (corners[-1][0], corners[-1][1]), (corners[0][0], corners[0][1]), color_empty, 5)

                img_show = cv2.line(
                    img_show, (454, 0), (454, 550), color_empty, 10)
            else:
                img_show = cv2.polylines(
                    img_show, [corners], False, color_empty, 20)
        return img_show

    def draw_map_with_tags(self):
        img_show = self.draw_map()
        for tag_id, pose in self.tag_history.items():
            pos = (int(pose['trans'][0]*100), int(pose['trans'][1]*100))
            img_show = cv2.circle(img_show, pos, 5, (255, 0, 0), -1)
            cv2.putText(img_show, str(tag_id), pos,
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0), 4)
        return img_show

    def find_empty_spots_(self):
        empty_spots = []
        for spot_id, corners in parking_spots.items():
            if spot_id in self.tag_history:
                corners = np.asarray(corners, np.float)
                v = (corners[0] + corners[-1]) - (corners[1] + corners[2])
                v = v / np.linalg.norm(v)
                # yaw = np.arccos(v[0]) if v[1] > 0 else -np.arccos(v[0])
                yaw = np.arccos(v[0])
                xy = [np.mean(corners[:, 0])/100.,
                      np.mean(corners[:, 1])/100., yaw]
                empty_spots.append(xy)
        return np.array(empty_spots)

    def find_empty_spots(self):
        empty_spots = {}
        for spot_id, corners in parking_spots.items():
            if spot_id in self.tag_history:
                corners = np.asarray(corners, np.float)
                v = (corners[0] + corners[-1]) - (corners[1] + corners[2])
                v = v / np.linalg.norm(v)
                # yaw = np.arccos(v[0]) if v[1] > 0 else -np.arccos(v[0])
                yaw = np.arccos(v[0])
                xy = [np.mean(corners[:, 0])/100.,
                      np.mean(corners[:, 1])/100., yaw]
                empty_spots[spot_id] = np.array(xy)
        return empty_spots


class EKF:
    def __init__(self, initial_pose,
                 xy_obs_noise_std=1.0,
                 initial_yaw_std=np.pi,
                 forward_velocity_noise_std=1.0,
                 yaw_rate_noise_std=0.17):

        # initial state
        self.initial_yaw_std = np.pi
        self.initial_yaw = initial_pose[2]
        self.x = initial_pose.copy()

        # covariance for initial state estimation error (Sigma_0)
        self.P = np.array([
            [xy_obs_noise_std ** 2., 0., 0.],
            [0., xy_obs_noise_std ** 2., 0.],
            [0., 0., initial_yaw_std ** 2.]
        ])

        # Prepare measurement error covariance Q
        self.Q = np.array([
            [xy_obs_noise_std ** 2., 0.],
            [0., xy_obs_noise_std ** 2.]
        ])

        # Prepare state transition noise covariance R
        self.R = np.array([
            [forward_velocity_noise_std ** 2., 0., 0.],
            [0., forward_velocity_noise_std ** 2., 0.],
            [0., 0., yaw_rate_noise_std ** 2.]
        ])

        # initialize Kalman filter
        self.kf = ExtendedKalmanFilter(self.x, self.P)

        # array to store estimated 2d pose [x, y, theta]
        self.mu_x = [self.x[0], self.x[0]+0.1]
        self.mu_y = [self.x[1], self.x[1]+0.1]
        self.mu_theta = [self.x[2], self.x[2]+0.1]

        # array to store estimated error variance of 2d pose
        self.var_x = [self.P[0, 0], self.P[0, 0]+0.01]
        self.var_y = [self.P[1, 1], self.P[1, 1]+0.01]
        self.var_theta = [self.P[2, 2], self.P[2, 2]+0.01]

        self.times = [time.time(), time.time()+0.01]
        self.pose_last = initial_pose.copy()
        self.obs_poses = [initial_pose.copy(), initial_pose.copy()]

    def apply(self, pose):
        t = time.time()
        dt = t - self.times[-1]

        # get control input `u = [v, omega] + noise`
        obs_forward_velocities = np.linalg.norm(pose[:2] - self.pose_last[:2]) / dt
        obs_yaw_rates = (pose[2] - self.pose_last[2]) / dt

        # mu_dt = self.times[-1] - self.times[-2]
        # if mu_dt < 0.0001:
        #     mu_dt = 0.0001
        # err = np.sqrt((self.mu_x[-1] - self.mu_x[-2])**2 + (self.mu_y[-1] - self.mu_y[-2])**2)
        # obs_forward_velocities = err / mu_dt
        # obs_yaw_rates = (self.mu_theta[-1] - self.mu_theta[-2]) / mu_dt

        u = np.array([
            obs_forward_velocities,
            obs_yaw_rates
        ])

        # propagate!
        R_ = self.R * (dt ** 2.)
        self.kf.propagate(u, dt, R_)

        # get measurement `z = [x, y] + noise`
        z = np.array([
            pose[0],
            pose[1]
        ])

        # update!
        self.kf.update(z, self.Q)

        # save estimated state to analyze later
        self.mu_x.append(self.kf.x[0])
        self.mu_y.append(self.kf.x[1])
        self.mu_theta.append(normalize_angles(self.kf.x[2]))

        # save estimated variance to analyze later
        self.var_x.append(self.kf.P[0, 0])
        self.var_y.append(self.kf.P[1, 1])
        self.var_theta.append(self.kf.P[2, 2])

        # self.pose_last = pose.copy()
        self.pose_last = np.array([self.mu_x[-1], self.mu_y[-1], self.mu_theta[-1]])
        self.times.append(t)
        self.obs_poses.append(pose.copy())

        return np.array([self.mu_x[-1], self.mu_y[-1], self.mu_theta[-1]])

    def show_results(self, ax=plt.gca()):
        obs_poses = np.array(self.obs_poses)[10:]
        mu_x = np.array(self.mu_x)[10:]
        mu_y = np.array(self.mu_y)[10:]

        ax.set_title("EKF: Position Evaluation")
        ax.plot(obs_poses[:, 0], obs_poses[:, 1], lw=1,
                color='b', markersize=3, alpha=0.4, label='observed')
        ax.plot(mu_x, mu_y, lw=1, label='estimated', color='r')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.legend()
        ax.axis("equal")
