# Capture at most 6 images in the video "outpy.avi" with 20 frame capture interval.

import cv2

vc = cv2.VideoCapture(0)  # read .avi file
c = 1

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

timeF = 30  # set frame capture interval
count = 0
while rval and count<8:  # read frame in loop
    rval, frame = vc.read()
    if c % timeF == 0:  # save each timeF frame
        cv2.imwrite('image' + str(count+1) + '.jpg', frame)  # save as .jpg
        count=count+1
    c = c + 1
    cv2.waitKey(1)
vc.release()
