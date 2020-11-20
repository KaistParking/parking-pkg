import cv2

from tag_recognition.watcher import Watcher

cap = cv2.VideoCapture(0)

_, img_color = cap.read()
img_h, img_w = img_color.shape[:2]
watcher = Watcher(img_size=(img_w, img_h), tag_size=0.16)

while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break

    ret, img_color = cap.read()
    if ret is None:
        break

    watcher.watch(img_color)

    img_tags = watcher.draw_tags()
    img_map = watcher.draw_map_with_tags()

    cv2.imshow('tags', img_tags)
    cv2.imshow('map', img_map)

print("end")
