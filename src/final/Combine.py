from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import cv2

BOOK_PERIMETER_INCH = 40
SHRINK_PEOPLE_HEIGHT = 0.9

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

MIN_MATCH_COUNT = 30
MIN_MATCH_COUNT_BOOK = 45

detector1 = cv2.xfeatures2d.SIFT_create(1500)
detector2 = cv2.xfeatures2d.SIFT_create(1500)

FLANN_INDEX_KDITREE = 0
flannParam = dict(algorithm=FLANN_INDEX_KDITREE, tree=5)   # FLANN based Matcher
flann = cv2.FlannBasedMatcher(flannParam, {})
flannParam_book = dict(algorithm=FLANN_INDEX_KDITREE, tree=5)
flann_book = cv2.FlannBasedMatcher(flannParam_book, {})

trainImg = cv2.imread("TrainingData/love.png", 0)
trainKP, trainDesc = detector1.detectAndCompute(trainImg, None)

trainImg_book = cv2.imread("TrainingData/book.png", 0)
trainKP_book, trainDesc_book = detector2.detectAndCompute(trainImg_book, None)

# Initial text
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale_large = 2.5
fontScale_medium = 1.5
# fontColor = (0, 255, 255)
lineType = 3


def detect_logo(partimg):
    queryKP, queryDesc = detector2.detectAndCompute(partimg, None)
    matches = flann_book.knnMatch(queryDesc, trainDesc, k=2)

    goodMatch = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            goodMatch.append(m)
    if len(goodMatch) > MIN_MATCH_COUNT:
        return True
    else:
        return False


def draw_book(QueryImgBGR):
    QueryImg = cv2.cvtColor(QueryImgBGR, cv2.COLOR_BGR2GRAY)
    queryKP, queryDesc = detector2.detectAndCompute(QueryImg, None)
    matches = flann_book.knnMatch(queryDesc, trainDesc_book, k=2)
    average_x = 0
    min_h = 1000
    max_h = 0
    perimeter_px = 0
    # book_height = 0
    goodMatch = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            goodMatch.append(m)
    if len(goodMatch) > MIN_MATCH_COUNT_BOOK:
        tp = []
        qp = []
        for m in goodMatch:
            tp.append(trainKP_book[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp, qp = np.float32((tp, qp))
        H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
        h, w = trainImg_book.shape
        trainBorder = np.float32([[[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]])
        queryBorder = cv2.perspectiveTransform(trainBorder, H)
        for item in queryBorder[0]:
            average_x += item[0]
            min_h = min(item[1], min_h)
            max_h = max(item[1], max_h)
        average_x = average_x / 4
        cv2.polylines(QueryImgBGR, [np.int32(queryBorder)], True, (0, 255, 0), 5)
        perimeter_px = compute_perimeter(queryBorder[0])
    return max(0, perimeter_px), average_x  # 返回这玩意的px周长，以及它所在的中心的x左边坐标


def compute_perimeter(queryBorder):
    array = np.int32(queryBorder)
    perimeter = 0
    perimeter += ((array[0][0] - array[1][0]) ** 2 + (array[0][1] - array[1][1]) ** 2) ** 0.5
    perimeter += ((array[1][0] - array[2][0]) ** 2 + (array[1][1] - array[2][1]) ** 2) ** 0.5
    perimeter += ((array[2][0] - array[3][0]) ** 2 + (array[2][1] - array[3][1]) ** 2) ** 0.5
    perimeter += ((array[0][0] - array[3][0]) ** 2 + (array[0][1] - array[3][1]) ** 2) ** 0.5
    # print(array)
    return perimeter


if __name__ == '__main__':
    while cap.isOpened():
        ret, image = cap.read()
        if ret:
            orig = image.copy()
            actual_height = 0
            book_perimeter_px, book_x = draw_book(orig)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            scale = image.shape[1] / shrink_w
            image = imutils.resize(image, width=min(shrink_w, image.shape[1]))
            # detect people in the image
            (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), scale=1.025)

            # apply non-maxima suppression to the bounding boxes using a
            # fairly large overlap threshold to try to maintain overlapping
            # boxes that are still people
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            people = non_max_suppression(rects, probs=None, overlapThresh=0.65)
            # transfer to original coordination
            people = [[int(c * scale) for c in person] for person in people]

            for (xA, yA, xB, yB) in people:
                person_img = orig[yA:yB, xA:xB]
                gray_person_img = gray[yA:yB, xA:xB]
                if detect_logo(gray_person_img):  # has logo person
                    cv2.rectangle(orig, (xA, yA), (xB, yB), (0, 255, 0), 5)
                    if book_perimeter_px > 0:
                        actual_height = (yB - yA) * SHRINK_PEOPLE_HEIGHT / book_perimeter_px * BOOK_PERIMETER_INCH
                        cv2.putText(orig, "{0:.2f}inch".format(actual_height),
                                    (xA, yA - 20),
                                    font,
                                    fontScale_large,
                                    (0, 255, 0),
                                    lineType)
                    faces = face_cascade.detectMultiScale(gray_person_img, 1.025, 5)
                    for (x, y, w, h) in faces:
                        # print("people height(px) = ", yB-yA)
                        # print("Logo is in x=", logo_x, ",  x range of people is  ", xA, " and ", xB)
                        cv2.rectangle(person_img, (x, y), (x + w, y + h), (0, 0, 255), 5)  # draw face
                        roi_gray = gray_person_img[y:y + int(2 * h / 3), x:x + w]
                        roi_color = person_img[y:y + h, x:x + w]
                        large_roi_gray = imutils.resize(roi_gray, width=w * 2)
                        eyes = eye_cascade.detectMultiScale(large_roi_gray, 1.025)
                        for (ex, ey, ew, eh) in eyes:
                            cv2.rectangle(roi_color, (int(ex / 2), int(ey / 2)),
                                          (int(ex / 2 + ew / 2), int(ey / 2 + eh / 2)), (0, 255, 0), 5)  # draw eyes
                        break
                else:  # no logo person
                    cv2.rectangle(orig, (xA, yA), (xB, yB), (255, 255, 255), 5)
                    if book_perimeter_px > 0:
                        actual_height = (yB - yA) * SHRINK_PEOPLE_HEIGHT / book_perimeter_px * BOOK_PERIMETER_INCH
                        cv2.putText(orig, "{0:.2f}inch".format(actual_height),
                                    (xA, yA - 20),
                                    font,
                                    fontScale_medium,
                                    (255, 255, 255),
                                    lineType)

            # show the output images
            cv2.imshow("Detector", orig)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
