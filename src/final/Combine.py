from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import cv2

# define cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)

# Get the width and height of frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

shrink_w = 400

# 以下是detect logo的各个定义
MIN_MATCH_COUNT = 25

detector = cv2.xfeatures2d.SIFT_create(1500)

FLANN_INDEX_KDITREE = 0
flannParam = dict(algorithm=FLANN_INDEX_KDITREE, tree=5)
flann = cv2.FlannBasedMatcher(flannParam, {})

trainImg = cv2.imread("TrainingData/far_train.png", 0)
trainKP, trainDesc = detector.detectAndCompute(trainImg, None)


def get_logo_x(QueryImgBGR):  # 获得logo所在位置的x坐标
    QueryImg = cv2.cvtColor(QueryImgBGR, cv2.COLOR_BGR2GRAY)
    queryKP, queryDesc = detector.detectAndCompute(QueryImg, None)
    matches = flann.knnMatch(queryDesc, trainDesc, k=2)
    average_x = 0
    goodMatch = []
    for m, n in matches:
        if (m.distance < 0.75 * n.distance):
            goodMatch.append(m)
    if (len(goodMatch) > MIN_MATCH_COUNT):
        tp = []
        qp = []
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp, qp = np.float32((tp, qp))
        H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
        h, w = trainImg.shape
        trainBorder = np.float32([[[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]])
        queryBorder = cv2.perspectiveTransform(trainBorder, H)
        # todo 加入project.py detect到包括此x坐标的方框，显示出来，然后识别出来脸等
        for item in queryBorder[0]:
            average_x += item[0]
        average_x = average_x / 4
    # else:
    #     print("Not Enough match found- %d/%d" % (len(goodMatch), MIN_MATCH_COUNT))
    return average_x


if __name__ == '__main__':
    # config real-time playback
    while cap.isOpened():
        ret, image = cap.read()
        if ret:
            orig = image.copy()
            logo_x = get_logo_x(orig)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            scale = image.shape[1] / shrink_w
            image = imutils.resize(image, width=min(shrink_w, image.shape[1]))

            # detect people in the image
            (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), scale=1.025, padding=(16, 16))

            # apply non-maxima suppression to the bounding boxes using a
            # fairly large overlap threshold to try to maintain overlapping
            # boxes that are still people
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            people = non_max_suppression(rects, probs=None, overlapThresh=0.65)

            # transfer to original coordination
            people = [[int(c * scale) for c in person] for person in people]

            for (xA, yA, xB, yB) in people:
                # draw bounding boxes for the person
                if logo_x > xA and logo_x < xB:
                    cv2.rectangle(orig, (xA, yA), (xB, yB), (0, 255, 0), 5)  # 画人 绿
                    # print("Logo is in x=", logo_x, ",  x range of people is  ", xA, " and ", xB)
                    # face detection
                    person_img = orig[yA:yB, xA:xB]
                    gray_person_img = gray[yA:yB, xA:xB]
                    faces = face_cascade.detectMultiScale(gray_person_img, 1.025, 5)
                    for (x, y, w, h) in faces:
                        # draw face bounding box
                        cv2.rectangle(person_img, (x, y), (x + w, y + h), (0, 0, 255), 5)  # 画脸  蓝

                        roi_gray = gray_person_img[y:y + h, x:x + w]
                        roi_color = person_img[y:y + h, x:x + w]
                        eyes = eye_cascade.detectMultiScale(roi_gray, 1.025)
                        for (ex, ey, ew, eh) in eyes:
                            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)  # 画眼睛  绿
                        break

            # show the output images
            cv2.imshow("Detector", orig)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
                break
        else:
            break

    # out.release()
    cap.release()
    cv2.destroyAllWindows()
