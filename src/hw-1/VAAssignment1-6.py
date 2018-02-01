# Add and remove S&P noise in an image.

import cv2
import numpy as np


# add salt and pepper noise
def saltpepper(img, n):
    m = int((img.shape[0] * img.shape[1]) * n)
    for a in range(m):
        i = int(np.random.random() * img.shape[1])
        j = int(np.random.random() * img.shape[0])
        if img.ndim == 2:
            img[j, i] = 255
        elif img.ndim == 3:
            img[j, i, 0] = 255
            img[j, i, 1] = 255
            img[j, i, 2] = 255
    for b in range(m):
        i = int(np.random.random() * img.shape[1])
        j = int(np.random.random() * img.shape[0])
        if img.ndim == 2:
            img[j, i] = 0
        elif img.ndim == 3:
            img[j, i, 0] = 0
            img[j, i, 1] = 0
            img[j, i, 2] = 0
    return img


img = cv2.imread('image3.jpg')
saltImage = saltpepper(img, 0.02)  # Create salt&pepper noise image
cv2.imwrite('saltPepperImage.jpg', saltImage)
median_blur = cv2.medianBlur(saltImage, 5)  # remove salt and pepper
cv2.imshow("Afterblur", median_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
