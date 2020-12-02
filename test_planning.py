import numpy as np
import cv2
import matplotlib.pyplot as plt

import warnings
import sys
sys.path.append('./src')
warnings.filterwarnings(action='ignore')

try:
    from tag_recognition.watcher import Watcher
    from planning.planner import Planner
except ImportError:
    from src.tag_recognition.watcher import Watcher
    from src.planning.planner import Planner

# draw map
watcher = Watcher()
map_color = watcher.draw_map(color_full=(255, 255, 255), color_empty=(0, 0, 0))
map_planning = cv2.resize(map_color, dsize=(0, 0), fx=0.1, fy=0.1)
map_planning[np.where(map_planning != 255)] = 0

# planning (m)
planner = Planner(map_planning, meter_scale=0.01/0.1)
start = [4.1, 0.3, np.deg2rad(90)]
end = [1.3, 4.5, np.deg2rad(90)]
path = planner.plan_path(start, end)

planner.show_path()
plt.show()
