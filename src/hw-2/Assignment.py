from collections import deque
import numpy as np
import cv2
import time
import math

lower_white = np.array([40, 30, 190], dtype=np.uint8)
upper_white = np.array([100, 100, 255], dtype=np.uint8)
lower_green = np.array([40, 70, 70])
upper_green = np.array([80, 200, 200])
# initial list of tracking points
my_buffer = 300
total_distance = 0  # calculate total distance during the whole process
pts = deque(maxlen=my_buffer)
camera = cv2.VideoCapture('cs6327-a2.mp4')
size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
totaltime = camera.get(cv2.CAP_PROP_FRAME_COUNT) / camera.get(cv2.CAP_PROP_FPS)
time.sleep(2)

while True:
    (ret, frame) = camera.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    mask = cv2.medianBlur(mask, 9)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    median_blur = cv2.medianBlur(mask_green, 19)  # remove salt and pepper)
    # cv2.imshow("mask",mask)


    # detect contour
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]  # White mask contour
    (_, green_contours, _) = cv2.findContours(median_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # Green mask contour

    maxrec = max(green_contours, key=cv2.contourArea)
    rect = cv2.boundingRect(maxrec)

    center = None
    # if exist contour
    if len(cnts) > 0:
        # find the maximum contour within the green mask area
        index = -1
        while True:
            c = sorted(cnts, key=cv2.contourArea)[index]  # Traverse contour list from largest to smallest
            # decide its out circle
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if x > rect[0] and x < rect[0] + rect[2] and y > rect[1] and y < rect[1] + rect[3]:
                break
            else:
                index -= 1

        M = cv2.moments(c)
        # calculate center
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # draw the circle
        cv2.circle(frame, (int(x), int(y)), 12, (0, 255, 0), 2)
        cv2.circle(frame, center, 3, (0, 0, 255), -1)
        # add center to deque
        pts.appendleft(center)

    # initial the black image
    black_img = np.zeros([size[1], size[0], 3], dtype=np.uint8)
    black_img.fill(0)

    # traverse track points,draw paths with white color
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        # draw line
        cv2.line(frame, pts[i - 1], pts[i], (255, 255, 255), 2)
        cv2.line(black_img, pts[i - 1], pts[i], (255, 255, 255), 2)
        total_distance += math.sqrt(math.pow(pts[i][1] - pts[i - 1][1], 2) + math.pow(pts[i][0] - pts[i - 1][0], 2))

    cv2.imshow('Frame', frame)
    cv2.imwrite("path.jpg", black_img)

    k = cv2.waitKey(5) & 0xFF
    if k == 5:
        break

camera.release()
cv2.destroyAllWindows()

print("speed = ", total_distance / totaltime, "(pixel/second)")
