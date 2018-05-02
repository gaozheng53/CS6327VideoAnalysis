from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import cv2

# define cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)  # Capture video from camera

# Get the width and height of frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

shrink_w = 400
# config real-time playback
while cap.isOpened():
    ret, image = cap.read()
    if ret:

        orig = image.copy()
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
            # draw bounding boxes for the person
            cv2.rectangle(orig, (xA, yA), (xB, yB), (0, 255, 0), 2)

            # face detection
            person_img = orig[yA:yB, xA:xB]
            gray_person_img = gray[yA:yB, xA:xB]
            faces = face_cascade.detectMultiScale(gray_person_img, 1.025, 5)
            for (x, y, w, h) in faces:
                # draw face bounding box
                cv2.rectangle(person_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                roi_gray = gray_person_img[y:y + h, x:x + w]
                roi_color = person_img[y:y + int(h/2), x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.025)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
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
