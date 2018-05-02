import cv2
import numpy as np

MIN_MATCH_COUNT = 50

detector = cv2.xfeatures2d.SIFT_create(1500)

FLANN_INDEX_KDITREE = 0
flannParam = dict(algorithm=FLANN_INDEX_KDITREE, tree=5)
flann = cv2.FlannBasedMatcher(flannParam, {})

trainImg = cv2.imread("TrainingData/1(13).jpg", 0)
trainKP, trainDesc = detector.detectAndCompute(trainImg, None)

cam = cv2.VideoCapture(0)
while True:
    ret, QueryImgBGR = cam.read()
    QueryImg = cv2.cvtColor(QueryImgBGR, cv2.COLOR_BGR2GRAY)
    queryKP, queryDesc = detector.detectAndCompute(QueryImg, None)
    matches = flann.knnMatch(queryDesc, trainDesc, k=2)

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
        print(queryBorder)
        # todo 加入遍历数组得到最后平均的x坐标，再加入project.py detect到包括此x坐标的方框，显示出来，然后识别出来脸等
        cv2.polylines(QueryImgBGR, [np.int32(queryBorder)], True, (0, 255, 0), 5)
    else:
        print("Not Enough match found- %d/%d" % (len(goodMatch), MIN_MATCH_COUNT))
    cv2.imshow('result', QueryImgBGR)
    if cv2.waitKey(10) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
