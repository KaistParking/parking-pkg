import numpy as np
import cv2
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

from pupil_apriltags import Detector

# c920 params
fx = 646.47368
fy = 645.33399
cx = 298.07274
cy = 231.63218
param = [fx, fy, cx, cy]

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

tag_rot_base = np.array([1, 1, 1])
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

# h_max = np.max([np.asarray(val)[:, 1] for val in parking_spots.values()])
# for tag_id in parking_spots:
#     contours = parking_spots[tag_id]
#     for i in range(4):
#         parking_spots[tag_id][i][1] = h_max - parking_spots[tag_id][i][1]

# for tag_id in tag_status:
#     trans = tag_status[tag_id]['trans']
#     tag_status[tag_id]['trans'][1] = h_max//100 - trans[1]


class Watcher:
    def __init__(self, img_size=(1920, 1080), tag_size=0.15):
        self.map_w = np.max([np.asarray(val)[:, 0]
                             for val in parking_spots.values()])
        self.map_h = np.max([np.asarray(val)[:, 1]
                             for val in parking_spots.values()])

        self.detector = Detector(families='tag36h11',
                                 nthreads=4,
                                 quad_decimate=1.0,
                                 quad_sigma=0.0,
                                 refine_edges=1,
                                 decode_sharpening=0.25,
                                 debug=0)

        self.img_size = img_size
        self.tag_size = tag_size

        self.cam_trans = None
        self.cam_rot = None

        self.img = None
        self.tags = None
        self.tag_poses = None
        self.tag_history = {}

    def find_camera_frame(self):
        cam_translation = []
        cam_rotation = []

        for tag in self.tags:
            # filter err & tag-ids
            # if tag.pose_err > 0.000001:
            #     continue
            if tag.tag_id not in tag_status:
                continue

            cam_translation.append(
                tag_status[tag.tag_id]['trans'] - tag.pose_t[:, 0])
            cam_rotation.append(np.matmul(np.linalg.inv(
                tag.pose_R), tag_status[tag.tag_id]['rot']))

        if len(cam_translation) == 0:
            return None, None

        trans = np.mean(cam_translation, axis=0)
        rot = np.sum(cam_rotation, axis=0)
        rot = rot / np.linalg.norm(rot)
        # trans = cam_translation[0]
        # rot = cam_rotation[0]
        # return np.array([0., 0., 0.]), np.array([[1,0,0], [0,1,0], [0,0,1]])
        return trans, rot

    def find_camera_frame2(self):
        cam_translation = []
        cam_rotation = []
        q_diff = Quaternion(axis=[1, 1, 1], angle=0.0)

        for tag in self.tags:
            if tag.tag_id not in tag_status:
                continue

            # world to camera
            q_camera = Quaternion(matrix=tag.pose_R)
            q_world = Quaternion(tag_status[tag.tag_id]['rot'])
            q_diff = q_camera - q_world
            cam_in_world = tag_status[tag.tag_id]['rot'] - \
                q_diff.rotate(tag.pose_t)
            cam_translation.append(cam_in_world)
            cam_rotation.append(q_diff)

        if len(cam_translation) == 0:
            return None, None

        trans_mean = np.mean(cam_translation, axis=0)
        rot_mean = q_diff[0]
        return trans_mean, rot_mean

    def calculate_tag_pose(self):
        results = {}
        for tag in self.tags:
            trans = tag.pose_t[:, 0] + self.cam_trans
            rot = np.matmul(tag.pose_R, self.cam_rot)
            results[tag.tag_id] = {'trans': trans, 'rot': rot}
            self.tag_history[tag.tag_id] = {'trans': trans, 'rot': rot}
        return results

    def watch(self, img_color):
        # detect tags
        self.img = img_color.copy()
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, self.img_size)
        self.tags = self.detector.detect(
            img_gray, estimate_tag_pose=True, camera_params=param, tag_size=self.tag_size)

        # find camera pose
        cam_trans, cam_rot = self.find_camera_frame()
        if cam_trans is None:
            return None
        self.cam_trans, self.cam_rot = cam_trans, cam_rot

        # calculate tag poses
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

    def draw_map(self, color_full=(255, 150, 150), color_empty=(200, 0, 255)):
        img_show = 255 * \
            np.ones(shape=(self.map_h, self.map_w, 3), dtype=np.uint8)
        for spot_id, corners in parking_spots.items():
            corners = np.asarray(corners, np.int64)
            if spot_id not in self.tag_history:
                img_show = cv2.fillPoly(img_show, [corners], color_full)
                img_show = cv2.polylines(
                    img_show, [corners], False, color_empty, 20)
            else:
                img_show = cv2.polylines(
                    img_show, [corners], False, color_empty, 20)
        return img_show

    def draw_map_with_tags(self):
        img_show = self.draw_map()
        for tag_id, pose in self.tag_history.items():
            pos = (int(pose['trans'][0]*100), int(pose['trans'][1]*100))
            img_show = cv2.circle(img_show, pos, 10, (150, 150, 250), -1)
            cv2.putText(img_show, str(tag_id), pos,
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0), 4)
        if self.cam_trans is not None:
            img_show = cv2.circle(img_show, (int(
                self.cam_trans[0]*100), int(self.cam_trans[1]*100)), 20, (255, 0, 0), -1)
        return img_show

    def find_empty_spots(self):
        empty_spots = []
        for spot_id, corners in parking_spots.items():
            if spot_id in self.tag_history:
                corners = np.asarray(corners, np.float)
                v = (corners[0] + corners[-1]) - (corners[1] + corners[2])
                v = v / np.linalg.norm(v)
                yaw = np.arccos(v[0]) if v[1] > 0 else -np.arccos(v[0])
                xy = [np.mean(corners[:, 0]), np.mean(corners[:, 1]), yaw]
                empty_spots.append(xy)
        return np.array(empty_spots)


class Watcher2:
    def __init__(self, img_size=(1920, 1080), tag_size=0.15):
        self.map_w = np.max([np.asarray(val)[:, 0]
                             for val in parking_spots.values()])
        self.map_h = np.max([np.asarray(val)[:, 1]
                             for val in parking_spots.values()])

        self.detector = Detector(families='tag36h11',
                                 nthreads=4,
                                 quad_decimate=1.0,
                                 quad_sigma=0.0,
                                 refine_edges=1,
                                 decode_sharpening=0.25,
                                 debug=0)

        self.img_size = img_size
        self.tag_size = tag_size

        self.cam_trans = None
        self.cam_rot = None

        self.img = None
        self.tags = None
        self.tag_poses = None
        self.tag_history = {}

    def find_camera_frame(self):
        cam_translation = []
        cam_rotation = []
        q_diff = Quaternion(axis=[1, 1, 1], angle=0.0)

        for tag in self.tags:
            if tag.tag_id not in tag_status:
                continue

            # world to camera
            q_base = Quaternion(axis=[1, 1, 1], angle=0.0)
            q_moved = Quaternion(axis=np.matmul(
                tag.pose_R, [1, 1, 1]), angle=0.0)
            q_camera = q_moved - q_base
            q_world = Quaternion(axis=tag_status[tag.tag_id]['rot'], angle=0.)
            # q_diff = q_world - q_camera
            q_diff = q_camera - q_world
            cam_in_world = tag_status[tag.tag_id]['trans'] - \
                q_diff.rotate(tag.pose_t)
            cam_translation.append(cam_in_world)
            cam_rotation.append(q_diff)

        if len(cam_translation) == 0:
            return None, None

        trans_mean = np.mean(cam_translation, axis=0)
        rot_mean = cam_rotation[0]
        return trans_mean, rot_mean

    def calculate_tag_pose(self):
        results = {}
        for tag in self.tags:
            q_base = Quaternion(axis=[1, 1, 1], angle=0.0)
            q_moved = Quaternion(axis=np.matmul(
                tag.pose_R, [1, 1, 1]), angle=0.0)
            q_camera = q_moved - q_base

            trans_ = self.cam_rot.rotate(tag.pose_t[:, 0]) + self.cam_trans
            rot_ = q_camera - self.cam_rot
            results[tag.tag_id] = {'trans': trans_, 'rot': rot_.axis}
            self.tag_history[tag.tag_id] = {'trans': trans_, 'rot': rot_.axis}
        return results

    def watch(self, img_color):
        # detect tags
        self.img = img_color.copy()
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, self.img_size)
        self.tags = self.detector.detect(
            img_gray, estimate_tag_pose=True, camera_params=param, tag_size=self.tag_size)

        # find camera pose
        cam_trans, cam_rot = self.find_camera_frame()
        if cam_trans is None:
            return None
        self.cam_trans, self.cam_rot = cam_trans, cam_rot

        # calculate tag poses
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

    def draw_map(self, color_full=(255, 150, 150), color_empty=(200, 0, 255)):
        img_show = 255 * \
            np.ones(shape=(self.map_h, self.map_w, 3), dtype=np.uint8)
        for spot_id, corners in parking_spots.items():
            corners = np.asarray(corners, np.int64)
            if spot_id not in self.tag_history:
                img_show = cv2.fillPoly(img_show, [corners], color_full)
                img_show = cv2.polylines(
                    img_show, [corners], False, color_empty, 20)
            else:
                img_show = cv2.polylines(
                    img_show, [corners], False, color_empty, 20)
        return img_show

    def draw_map_with_tags(self):
        img_show = self.draw_map()
        for tag_id, pose in self.tag_history.items():
            pos = (int(pose['trans'][0]*100), int(pose['trans'][1]*100))
            img_show = cv2.circle(img_show, pos, 10, (150, 150, 250), -1)
            cv2.putText(img_show, str(tag_id), pos,
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0), 4)
        if self.cam_trans is not None:
            img_show = cv2.circle(img_show, (int(
                self.cam_trans[0]*100), int(self.cam_trans[1]*100)), 20, (255, 0, 0), -1)
        return img_show

    def find_empty_spots(self):
        empty_spots = []
        for spot_id, corners in parking_spots.items():
            if spot_id in self.tag_history:
                corners = np.asarray(corners, np.float)
                v = (corners[0] + corners[-1]) - (corners[1] + corners[2])
                v = v / np.linalg.norm(v)
                yaw = np.arccos(v[0]) if v[1] > 0 else -np.arccos(v[0])
                xy = [np.mean(corners[:, 0]), np.mean(corners[:, 1]), yaw]
                empty_spots.append(xy)
        return np.array(empty_spots)


class Watcher3:
    def __init__(self, img_size=(1920, 1080), tag_size=0.15):
        self.map_w = np.max([np.asarray(val)[:, 0]
                             for val in parking_spots.values()])
        self.map_h = np.max([np.asarray(val)[:, 1]
                             for val in parking_spots.values()])

        self.detector = Detector(families='tag36h11',
                                 nthreads=4,
                                 quad_decimate=1.0,
                                 quad_sigma=0.0,
                                 refine_edges=1,
                                 decode_sharpening=0.25,
                                 debug=0)

        self.img_size = img_size
        self.tag_size = tag_size

        self.cam_trans = None
        self.cam_rot = None

        self.img = None
        self.tags = None
        self.tag_poses = None
        self.tag_history = {}

    def find_camera_frame(self):
        cam_translation = []
        cam_rotation = []

        for tag in self.tags:
            if tag.tag_id not in tag_status:
                continue
            # world to camera
            r_cam = R.from_matrix(tag.pose_R)
            r_world = R.from_rotvec([0, 0, 0])

            # rot_world_to_cam = r_cam.inv() * \
            #     R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]
            #                 ) * R.from_euler('x', 180, degrees=True)
            rot_world_to_cam = r_cam.inv() * R.from_euler('x', 180, degrees=True)
            # world_coor = tag_status[tag.tag_id]['trans'] - \
            #     rot_world_to_cam.apply(tag.pose_t[:, 0])

            vec = tag.pose_t[:, 0]
            vec[0] = -vec[0]
            world_coor = r_cam.apply(vec)

            cam_translation.append(world_coor)
            cam_rotation.append(rot_world_to_cam)
        if len(cam_translation) == 0:
            return None, None

        trans_mean = np.mean(cam_translation, axis=0)
        rot_mean = cam_rotation[0]
        return trans_mean, rot_mean

    def calculate_tag_pose(self):
        results = {}
        for tag in self.tags:
            r_cam = R.from_matrix(tag.pose_R)
            trans_ = self.cam_rot.apply(tag.pose_t[:, 0]) + self.cam_trans
            rot_ = r_cam.as_rotvec()
            results[tag.tag_id] = {'trans': trans_, 'rot': rot_}
            self.tag_history[tag.tag_id] = {'trans': trans_, 'rot': rot_}
        return results

    def watch(self, img_color):
        # detect tags
        self.img = img_color.copy()
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, self.img_size)
        self.tags = self.detector.detect(
            img_gray, estimate_tag_pose=True, camera_params=param, tag_size=self.tag_size)

        # find camera pose
        cam_trans, cam_rot = self.find_camera_frame()
        if cam_trans is None:
            return None
        self.cam_trans, self.cam_rot = cam_trans, cam_rot

        # calculate tag poses
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

    def draw_map(self, color_full=(255, 150, 150), color_empty=(200, 0, 255)):
        img_show = 255 * \
            np.ones(shape=(self.map_h, self.map_w, 3), dtype=np.uint8)
        for spot_id, corners in parking_spots.items():
            corners = np.asarray(corners, np.int64)
            if spot_id not in self.tag_history:
                img_show = cv2.fillPoly(img_show, [corners], color_full)
                img_show = cv2.polylines(
                    img_show, [corners], False, color_empty, 20)
            else:
                img_show = cv2.polylines(
                    img_show, [corners], False, color_empty, 20)
        return img_show

    def draw_map_with_tags(self):
        img_show = self.draw_map()
        for tag_id, pose in self.tag_history.items():
            pos = (int(pose['trans'][0]*100), int(pose['trans'][1]*100))
            img_show = cv2.circle(img_show, pos, 10, (150, 150, 250), -1)
            cv2.putText(img_show, str(tag_id), pos,
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0), 4)
        if self.cam_trans is not None:
            img_show = cv2.circle(img_show, (int(
                self.cam_trans[0]*100), int(self.cam_trans[1]*100)), 20, (255, 0, 0), -1)
        return img_show

    def find_empty_spots(self):
        empty_spots = []
        for spot_id, corners in parking_spots.items():
            if spot_id in self.tag_history:
                corners = np.asarray(corners, np.float)
                v = (corners[0] + corners[-1]) - (corners[1] + corners[2])
                v = v / np.linalg.norm(v)
                yaw = np.arccos(v[0]) if v[1] > 0 else -np.arccos(v[0])
                xy = [np.mean(corners[:, 0]), np.mean(corners[:, 1]), yaw]
                empty_spots.append(xy)
        return np.array(empty_spots)

    def watch_camera(self, img_color):
        # detect tags
        self.img = img_color.copy()
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, self.img_size)
        self.tags = self.detector.detect(
            img_gray, estimate_tag_pose=True, camera_params=param, tag_size=self.tag_size)

        # find camera pose
        cam_trans, cam_rot = self.find_camera_frames()
        img_map = self.draw_map()
        if cam_trans is None:
            return img_map, None
        self.cam_trans, self.cam_rot = cam_trans, cam_rot
        return img_map, self.cam_trans

    def find_camera_frames(self):
        cam_translation = []
        cam_rotation = []

        for tag in self.tags:
            if tag.tag_id not in tag_status:
                continue
            # world to camera
            r_cam = R.from_matrix(tag.pose_R)
            r_world = R.from_rotvec([0, 0, 0])

            rot_world_to_cam = r_cam.inv() * \
                R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]
                            ) * R.from_euler('x', 180, degrees=True)
            world_coor = tag_status[tag.tag_id]['trans'] - \
                rot_world_to_cam.apply(tag.pose_t[:, 0])

            cam_translation.append(world_coor)
            cam_rotation.append(rot_world_to_cam)

        if len(cam_translation) == 0:
            return None, None

        return np.array(cam_translation), np.array(cam_rotation)
