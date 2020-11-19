import numpy as np
import cv2

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

tag_status = {
    1:  {'id': 1,   'trans': np.array([3.091, 0.80, 0]),    'rot': np.array([0, -1, 0])},
    2:  {'id': 2,   'trans': np.array([2.212, 0.734, 0]),   'rot': np.array([0, -1, 0])},
    3:  {'id': 3,   'trans': np.array([1.355, 0.78, 0]),    'rot': np.array([0, -1, 0])},
    4:  {'id': 4,   'trans': np.array([0.432, 0.838, 0]),   'rot': np.array([0, -1, 0])},
    5:  {'id': 5,   'trans': np.array([4.077, 4.90, 0]),    'rot': np.array([0, -1, 0])},
    6:  {'id': 6,   'trans': np.array([3.172, 4.89, 0]),    'rot': np.array([0, -1, 0])},
    7:  {'id': 7,   'trans': np.array([2.269, 4.86, 0]),    'rot': np.array([0, -1, 0])},
    8:  {'id': 8,   'trans': np.array([1.368, 4.88, 0]),    'rot': np.array([0, -1, 0])},
    9:  {'id': 9,   'trans': np.array([0.495, 4.81, 0]),    'rot': np.array([0, -1, 0])},
    10: {'id': 10,  'trans': np.array([2.68, 1.70, 0]),     'rot': np.array([0, -1, 0])},
    11: {'id': 11,  'trans': np.array([1.797, 1.70, 0]),    'rot': np.array([0, -1, 0])},
    12: {'id': 12,  'trans': np.array([0.905, 1.69, 0]),    'rot': np.array([0, -1, 0])},
    13: {'id': 13,  'trans': np.array([0.923, 3.87, 0]),    'rot': np.array([0, -1, 0])},
    14: {'id': 14,  'trans': np.array([1.802, 3.89, 0]),    'rot': np.array([0, -1, 0])},
    15: {'id': 15,  'trans': np.array([2.713, 3.90, 0]),    'rot': np.array([0, -1, 0])},
    16: {'id': 16,  'trans': np.array([3.602, 3.89, 0]),    'rot': np.array([0, -1, 0])},
}

for tag_id in parking_spots:
    contours = parking_spots[tag_id]
    for i in range(4):
        parking_spots[tag_id][i][1] = 600 - parking_spots[tag_id][i][1]

for tag_id in tag_status:
    trans = tag_status[tag_id]['trans']
    tag_status[tag_id]['trans'][1] = 6 - trans[1]


class Watcher:
    def __init__(self, map_size=(450, 554), img_size=(1920, 1080), tag_size=0.15):
        self.map_w = map_size[0]
        self.map_h = map_size[1]

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
            if tag.pose_err > 0.000001:
                continue
            if tag.tag_id not in tag_status:
                continue

            cam_translation.append(
                tag_status[tag.tag_id]['trans'] - tag.pose_t[:, 0])
            cam_rotation.append(np.matmul(np.linalg.inv(
                tag.pose_R), tag_status[tag.tag_id]['rot']))

        if len(cam_translation) == 0:
            return None, None

        # trans = np.mean(cam_translation, axis=0)
        # rot = np.sum(cam_rotation, axis=0)
        # rot = rot / np.linalg.norm(rot)
        trans = cam_translation[0]
        rot = cam_rotation[0]
        return trans, rot

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
                    img_show, [corners], False, color_empty, 10)
            else:
                img_show = cv2.polylines(
                    img_show, [corners], False, color_empty, 10)
        return img_show

    def draw_map_with_tags(self):
        img_show = self.draw_map()
        for tag_id, pose in self.tag_history.items():
            pos = (int(pose['trans'][0]*100), int(pose['trans'][1]*100))
            img_show = cv2.circle(img_show, pos, 20, (150, 150, 250), -1)
            cv2.putText(img_show, str(tag_id), pos,
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0), 4)
        return img_show
