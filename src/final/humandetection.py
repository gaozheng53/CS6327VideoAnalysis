# 完成detect到人和脸，眼睛

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# def inside(r, q):
#     rx, ry, rw, rh = r
#     qx, qy, qw, qh = q
#     return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness=3):
    for x, y, w, h in rects:
        if h > 400:   # detect到人的框大概是500-520的h
            # the HOG detector returns slightly larger rectangles than the real objects.
            # so we slightly shrink the rectangles to get a nicer output.
            pad_w, pad_h = int(0.15 * w), int(0.05 * h)  # 到时候shink height多一点就能准确测量对比身高一点了
            print("w = ", w, "  h = ", h)
            # if h > 200: # tune the result
            cv2.rectangle(img, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), thickness)
            ########### haar detection #############
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # colorpart = img[y + pad_h:y + h - pad_h, x + pad_w:x + w - pad_w]
            # graypart = gray[y + pad_h:y + h - pad_h, x + pad_w:x + w - pad_w]  # within the human rectangle region
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)   # 暂时先用gray吧，后面再改
            for (fx, fy, fw, fh) in faces:
                cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), thickness)
                roi_gray = gray[fy:fy + int(fh / 2), fx:fx + fw]  # 去除鼻子对眼睛detect的影响
                roi_color = img[fy:fy + int(fh / 2), fx:fx + fw]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), thickness)


if __name__ == '__main__':

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        found, w = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32),
                                        scale=1.05)  # found is tuple or ndarray
        draw_detections(frame, found)
        cv2.imshow('feed', frame)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
    cv2.destroyAllWindows()
