import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from tag_recognition.watcher import Watcher
from planning.planner import Planner
from control.controller import Controller

import warnings
warnings.filterwarnings(action='ignore')


results_dir = 'results'
results_idx = 0
while os.path.isdir(os.path.join(results_dir, str(results_idx))):
    results_idx += 1
results_dir = os.path.join(results_dir, str(results_idx))
os.mkdir(results_dir)


# draw map
watcher = Watcher()
map_color = watcher.draw_map(color_full=(255, 255, 255), color_empty=(0, 0, 0))
map_planning = cv2.resize(map_color, dsize=(0, 0), fx=0.1, fy=0.1)
map_planning[np.where(map_planning != 255)] = 0

# planning (m)
planner = Planner(map_planning, meter_scale=0.01/0.1)
start = [4.1, 0.3, np.deg2rad(90)]
end = [1.35, 4.2, np.deg2rad(90)]
path = planner.plan_path(start, end)
planner.show_path()
plt.show()

# control
map_shape = map_color.shape
car = Controller(path=path, map_color=map_color, map_size=(map_shape[1]/100, map_shape[0]/100))

plt.gcf().canvas.mpl_connect('key_release_event',
                             lambda event: [exit(0) if event.key == 'escape' else None])
img_idx = 0
while True:
    x0 = [car.state.x, car.state.y, car.state.yaw]
    path = None
    while path is None:
        path = planner.plan_path(x0, end)
    if len(path.x_list) < 5:
        break
    car.init_path(path)
    while not car.check_goal() and not (len(car.v) > 10 and np.mean(np.abs(car.v[-10:])) < 0.1):
        state = car.update_pose(pose=None, v=None)
        steer, accel = car.estimate_control(state=state)
        car.show(ax=plt.gca())

        plt.savefig('{}/{}.jpg'.format(results_dir, img_idx))
        img_idx += 1

        plt.pause(0.02)
        plt.gca().clear()

plt.show()
print('end')

fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
out = cv2.VideoWriter('results/{}.avi'.format(results_idx), fourcc, 10.0, (map_shape[1], map_shape[0]))
for i in tqdm(range(img_idx), desc='video'):
    img = cv2.imread('{}/{}.jpg'.format(results_dir, i))
    cv2.imshow('img', img)
    cv2.waitKey(100)
    out.write(img)
out.release()
print('video saved')
