# # Capture at most 6 images in the video "outpy.avi" with 20 frame capture interval.
#
# import cv2
#
# vc = cv2.VideoCapture('cs6327-a2.mp4')  # read .avi file
# c = 1
#
# if vc.isOpened():
#     rval, frame = vc.read()
# else:
#     rval = False
#
# timeF = 40 # set frame capture interval
# count = 0
# while rval and count < 8:  # read frame in loop
#     rval, frame = vc.read()
#     if c % timeF == 0:  # save each timeF frame
#         cv2.imwrite('image' + str(count + 1) + '.jpg', frame)  # save as .jpg
#         count = count + 1
#     c = c + 1
#     cv2.waitKey(1)
# vc.release()


import numpy as np
import cv2
import time
import math

lower_white = np.array([35, 70, 60], dtype=np.uint8)
upper_white = np.array([77, 255, 255], dtype=np.uint8)

frame = cv2.imread("image7.jpg")
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv, lower_white, upper_white)
mask1 = cv2.erode(mask1, None, iterations=5)
mask1 = cv2.dilate(mask1, None, iterations=7)
# mask1 = cv2.resize(mask1, (700, 456), interpolation=cv2.INTER_CUBIC)
# cv2.imshow("mask", mask1)
# cv2.waitKey(0)
mincenter = float("inf")  # infinite positive
minradius = float("inf")
_, contours, hierarchy = cv2.findContours(mask1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for contour in contours:
    (x, y), radius = cv2.minEnclosingCircle(contour)
    if radius < minradius:
        mincenter = (int(x), int(y))
        minradius = int(radius)

cv2.circle(frame, mincenter, minradius, (0, 255, 0), 2)

cv2.imshow("frame", frame)
cv2.waitKey(0)
