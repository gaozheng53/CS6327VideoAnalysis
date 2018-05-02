import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


def draw_detections(img, rects, thickness=3):
    for x, y, w, h in rects:
        # detect到人的框大概是500-520的h
        if h > 400:
            # the HOG detector returns slightly larger rectangles than the real objects.
            # so we slightly shrink the rectangles to get a nicer output.
            pad_w, pad_h = int(0.25 * w), int(0.08 * h)
            cv2.rectangle(img, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), thickness)
            ########### haar detection #############
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # colorpart = img[y + pad_h:y + h - pad_h, x + pad_w:x + w - pad_w]
            graypart = gray[y + pad_h:y + h - pad_h, x + pad_w:x + w - pad_w]  # within the human rectangle region

            faces = face_cascade.detectMultiScale(graypart, 1.3, 5)  # 暂时先用gray吧，后面再改
            for (fx, fy, fw, fh) in faces:
                cv2.rectangle(img, (x + pad_w + fx, y + pad_h + fy), (x + pad_w + fx + fw, y + pad_h + fy + fh),
                              (0, 0, 255), thickness)
                # cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), thickness)
                roi_gray = gray[y + pad_h + fy:y + pad_h + fy + int(fh / 2), x + pad_w + fx:x + pad_w + fx + fw]
                roi_color = img[y + pad_h + fy:y + pad_h + fy + int(fh / 2), x + pad_w + fx:x + pad_w + fx + fw]
                # roi_gray = gray[fy:fy + int(fh / 2), fx:fx + fw]  # 去除鼻子对detect的影响
                # roi_color = img[fy:fy + int(fh / 2), fx:fx + fw]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), thickness)


if __name__ == '__main__':

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        found, w = hog.detectMultiScale(frame, winStride=(4, 4),
                                        scale=1.05)  # found is tuple or ndarray
        draw_detections(frame, found)
        cv2.imshow('feed', frame)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
    cv2.destroyAllWindows()
