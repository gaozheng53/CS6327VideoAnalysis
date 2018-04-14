# import cv2
#
# for i in range(1,21):
#     image = cv2.imread("train/not/2("+str(i)+").jpg")
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # D3
#     cv2.imwrite("graytrain/not/"+str(i)+".jpg",gray_image)

import cv2

# for i in range(1,44):
#     image = cv2.imread("train/is/1("+str(i)+").jpg")
#     gray_image = cv2.resize(image,(64,64))
#     cv2.imwrite("resizetrain/is/"+str(i)+".jpg",gray_image)
#     image = cv2.imread("train/not/2(" + str(i) + ").jpg")
#     gray_image = cv2.resize(image, (64, 64))
#     cv2.imwrite("resizetrain/not/" + str(i) + ".jpg", gray_image)



image = cv2.imread("train/not/IMG_0130.jpg")
gray_image = cv2.resize(image, (64, 64))
cv2.imwrite("train/not/IMG_0130.jpg", gray_image)