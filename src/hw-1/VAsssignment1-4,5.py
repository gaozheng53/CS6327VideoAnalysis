# Do color transfer to detect orange and make it to white part with black background color.

import cv2
import numpy as np
import time


def transcolor(add, count):
    #    cap = cv2.VideoCapture(add)

    # set blue thresh
    lower_orange = np.array([0, 120, 100])
    upper_orange = np.array([150, 255, 255])

    #    while (1):
    # get a frame and show
    #        ret, frame = cap.read()
    frame = cv2.imread(add)
    # change to hsv model
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # get mask
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    cv2.imshow('Mask', mask)
    cv2.imwrite(str(count)+".jpg",mask)

    # if cv2.waitKey(1) & 0xFF == ord('q'):


#            break

#    cap.release()


t0 = time.time()
transcolor("image1.jpg", 1)
transcolor("image2.jpg", 2)
transcolor("image3.jpg", 3)
transcolor("image4.jpg", 4)
transcolor("image5.jpg", 5)
transcolor("image6.jpg", 6)

print("The average time is: " + str((time.time() - t0) / 6))
cv2.destroyAllWindows()
