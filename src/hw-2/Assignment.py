from collections import deque
import numpy as np
import cv2
import time

lower_white = np.array([40, 30, 190], dtype=np.uint8)
upper_white = np.array([100, 100, 255], dtype=np.uint8)
# initial list of tracking points
mybuffer = 300
totaldistance = 0  # calculate total distance during the whole process
pts = deque(maxlen=mybuffer)
camera = cv2.VideoCapture('cs6327-a2.mp4')
size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
time.sleep(2)

while True:
    (ret, frame) = camera.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # disregard some area
    hsv[0:1289][0:163] = [0, 0, 0]  # top
    hsv[0:1289][835:1100] = [0, 0, 0]  # bottom
    mask = cv2.inRange(hsv, lower_white, upper_white)
    mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=3)
    # cv2.imshow("mask", mask)

    # detect contour
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    center = None
    # if exist contour
    if len(cnts) > 0:
        # find the maximum contour
        c = max(cnts, key=cv2.contourArea)
        # decide its out circle
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        M = cv2.moments(c)
        # calculate center
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # only mark when r < 16
        if radius < 19:
            cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            # add center to deque
            pts.appendleft(center)

    blackimg = np.zeros([size[1], size[0], 3], dtype=np.uint8)
    blackimg.fill(0)

    # traverse track pints,draw path with small part
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        # draw line
        # if pts[i][0] - pts[i - 1][0] < 180 and pts[i][1] - pts[i - 1][1] < 180:
        cv2.line(frame, pts[i - 1], pts[i], (255, 255, 255), 2)
        cv2.line(blackimg, pts[i - 1], pts[i], (255, 255, 255), 2)
    cv2.imshow('Frame', frame)
    cv2.imwrite("path.jpg", blackimg)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

camera.release()
cv2.destroyAllWindows()
