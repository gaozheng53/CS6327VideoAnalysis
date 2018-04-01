import cv2
import numpy as np

time_duration = 1300

def generateColorizedImage(color_image, depth_image):
    # load the initial RGB image and its # of rows and columns
    color = cv2.imread(color_image)
    color_row, color_column = color.shape[:2] # third is channel
    # load the initial depth image and its # of rows and columns
    depth = cv2.imread(depth_image, cv2.IMREAD_ANYDEPTH)
    depth_row, depth_column = depth.shape
    blank_image = np.zeros((depth_row, depth_column, 3), np.uint8)
    # generate colorize image: combine with depth image and invIntrinsicIR
    for i in range(depth_row):
        for j in range(depth_column):
            matrix = np.matmul(np.array([j, i, 1]), invIntrinsicIR)
            if depth[i][j] > 0:
                matrix = np.multiply(matrix, depth[i][j])
                matrix = np.append(matrix, 1)
                matrix = np.matmul(matrix, transformationD_C)
                a = matrix[2]
                matrix = matrix / a
                matrix = matrix[0:3]
                matrix = np.matmul(matrix, intrinsicRGB)
                x_pix = int(matrix[0])
                y_pix = int(matrix[1])
                if(x_pix > 0 and x_pix < color_column and y_pix > 0 and y_pix < color_row):
                    blank_image[i][j] = color[y_pix][x_pix]
    return blank_image


def color_detect(image, res):
    lower = np.array([0,60,40])
    upper = np.array([16,255,255])
    img = cv2.imread(image)
    image = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.erode(mask, None, iterations = 2)
    mask = cv2.dilate(mask, None, iterations = 3)
    # cv2.imwrite(tmp, mask)

    cnt = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    c1 = max(cnt, key = cv2.contourArea)   # Get the maximum contour
    cnt.remove(c1)
    c2 = max(cnt, key = cv2.contourArea)
    ((x1, y1), radius1) = cv2.minEnclosingCircle(c1)
    ((x2, y2), radius2) = cv2.minEnclosingCircle(c2)
    cv2.rectangle(img, (int(x1 - radius1), int(y1 - radius1)),(int(x1 + radius1), int(y1 + radius1)),(0, 255, 0), 2)
    cv2.rectangle(img, (int(x2 - radius2), int(y2 - radius2)),(int(x2 + radius2), int(y2 + radius2)),(0, 0, 255), 2)

    cv2.imwrite(res, img)
    cv2.imshow(res, img)
    if (cv2.waitKey(0) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
    return (x1, y1), (x2, y2)


def compute_v(image1, image2, a1, b1, a2, b2):
    depth1 = cv2.imread(image1, cv2.IMREAD_ANYDEPTH)
    depth2 = cv2.imread(image2, cv2.IMREAD_ANYDEPTH)
    (x1, y1) = (int(b1), int(a1)) # convert to int
    (x2, y2) = (int(b2), int(a2))
    # coordination convert from 2d->3d
    matrix1 = np.matmul(np.array([x1, y1, 1]), invIntrinsicIR)

    matrix1 = np.multiply(matrix1, depth1[x1][y1])
    matrix2 = np.matmul(np.array([x2, y2, 1]), invIntrinsicIR)

    matrix2 = np.multiply(matrix2, depth2[x1][y1])
    distance = np.sqrt(np.square(matrix1[1] - matrix2[1]) + np.square(matrix1[2] - matrix2[2]))

    velocity = 1.0 * distance / (time_duration/1000)
    # print(velocity)
    return velocity


if __name__ == '__main__':
    # -------------Extract matrix elements from txt file------------------ #
    file = open('data/InvIntrinsicIR', 'r')
    invIntrinsicIR = []
    for line in file:
        invIntrinsicIR.append(line.strip().split(','))
    invIntrinsicIR = np.array(invIntrinsicIR).astype(np.float)

    file = open('data/IntrinsicRGB', 'r')
    intrinsicRGB = []
    for line in file:
        intrinsicRGB.append(line.strip().split(','))
    intrinsicRGB = np.array(intrinsicRGB).astype(np.float)

    file = open('data/TransformationD-C', 'r')
    transformationD_C = []
    for line in file:
        transformationD_C.append(line.strip().split(','))
    transformationD_C = np.array(transformationD_C).astype(np.float)
    file.close()

    # -------------Get colorize image------------------ #
    colorize_image1 = generateColorizedImage('data/color-63647317626781.png', 'data/depth-63647317626781.png' )
    colorize_image2 = generateColorizedImage('data/color-63647317628081.png', 'data/depth-63647317628081.png' )
    cv2.imwrite("colorize_1.jpg", colorize_image1)
    cv2.imwrite("colorize_2.jpg", colorize_image2)

    # -------------Detect ball based on color------------------ #
    (initial_x1, initial_y1), (initial_x2, initial_y2) = color_detect("colorize_1.jpg", "first_detect_result.jpg")
    (final_x1, final_y1), (final_x2, final_y2) = color_detect("colorize_2.jpg", "last_detect_result.jpg")

    # -------------Compute relative velocity------------------ #
    v1 = compute_v('data/depth-63647317626781.png', 'data/depth-63647317628081.png', initial_x1, initial_y1, final_x1, final_y1)
    v2 = compute_v('data/depth-63647317626781.png', 'data/depth-63647317628081.png', initial_x2, initial_y2, final_x2, final_y2)
    print('relative velocity =', v1 + v2, "mm/s")
