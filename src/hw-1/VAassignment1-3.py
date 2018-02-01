# Convert a image from BGR to HSV and show on the screen.

from cv2 import *

img = cv2.imread('image1.jpg', cv2.IMREAD_UNCHANGED)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # D3
im2 = cv2.resize(img, (700, 456), interpolation=cv2.INTER_CUBIC)
cv2.imshow('HSVimage', hsv)
# im2 = cv2.resize(hsv, (700, 456), interpolation=cv2.INTER_CUBIC)
# cv2.imshow('orangeimage', im2)
cv2.waitKey(0)
