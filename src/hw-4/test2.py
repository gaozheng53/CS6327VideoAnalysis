import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import glob
import time
import cv2
from scipy.ndimage.measurements import label


color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 15  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off


# a function to extract features from a list of images
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()  # Flatten
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def find_temoc(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]  # sub-sampling
    # ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YUV')
    ctrans_tosearch = img_tosearch
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 256
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    # nblocks_per_window = (window // pix_per_cell)-1

    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)

    bboxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (256, 256))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_stacked = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            test_features = X_scaler.transform(test_stacked)
            # test_features = scaler.transform(np.array(features).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
                bboxes.append(((int(xbox_left), int(ytop_draw + ystart)),
                               (int(xbox_left + win_draw), int(ytop_draw + win_draw + ystart))))

    return draw_img, bboxes


def apply_sliding_window(image, svc, X_scaler, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    #     apply to different scales
    bboxes = []
    ystart = 0
    ystop = 2000
    out_img, bboxes1 = find_temoc(image, ystart, ystop, 1.0, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 0
    ystop = 2000
    out_img, bboxes2 = find_temoc(out_img, ystart, ystop, 1.3, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 0
    ystop = 2000
    out_img, bboxes3 = find_temoc(out_img, ystart, ystop, 1.4, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 0
    ystop = 2000
    out_img, bboxes4 = find_temoc(out_img, ystart, ystop, 1.6, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 0
    ystop = 2000
    out_img, bboxes5 = find_temoc(out_img, ystart, ystop, 1.8, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 0
    ystop = 2000
    out_img, bboxes6 = find_temoc(out_img, ystart, ystop, 2.0, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 0
    ystop = 2000
    out_img, bboxes7 = find_temoc(out_img, ystart, ystop, 1.9, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 0
    ystop = 2000
    out_img, bboxes8 = find_temoc(out_img, ystart, ystop, 1.3, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 0
    ystop = 2000
    out_img, bboxes9 = find_temoc(out_img, ystart, ystop, 2.2, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 0
    ystop = 2000
    out_img, bboxes10 = find_temoc(out_img, ystart, ystop, 3.0, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                  spatial_size, hist_bins)
    bboxes.extend(bboxes1)
    bboxes.extend(bboxes2)
    bboxes.extend(bboxes3)
    bboxes.extend(bboxes4)
    bboxes.extend(bboxes5)
    bboxes.extend(bboxes6)
    bboxes.extend(bboxes7)
    bboxes.extend(bboxes8)
    bboxes.extend(bboxes9)
    bboxes.extend(bboxes10)

    return out_img, bboxes


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)


def find_temoc(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]  # sub-sampling
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YUV')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    # nblocks_per_window = (window // pix_per_cell)-1

    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)

    bboxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_stacked = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            test_features = X_scaler.transform(test_stacked)
            # test_features = scaler.transform(np.array(features).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
                bboxes.append(((int(xbox_left), int(ytop_draw + ystart)),
                               (int(xbox_left + win_draw), int(ytop_draw + win_draw + ystart))))

    return draw_img, bboxes


def apply_sliding_window(image, svc, X_scaler, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    #     apply to different scales
    bboxes = []
    ystart = 0
    ystop = 2000
    out_img, bboxes1 = find_temoc(image, ystart, ystop, 2.0, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 0
    ystop = 2000
    out_img, bboxes2 = find_temoc(out_img, ystart, ystop, 2.7, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 0
    ystop = 2000
    out_img, bboxes3 = find_temoc(out_img, ystart, ystop, 2.8, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 0
    ystop = 2000
    out_img, bboxes4 = find_temoc(out_img, ystart, ystop, 2.9, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 0
    ystop = 2000
    out_img, bboxes5 = find_temoc(out_img, ystart, ystop, 2.1, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 0
    ystop = 2000
    out_img, bboxes6 = find_temoc(out_img, ystart, ystop, 2.2, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 0
    ystop = 2000
    out_img, bboxes7 = find_temoc(out_img, ystart, ystop, 2.3, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 0
    ystop = 2000
    out_img, bboxes8 = find_temoc(out_img, ystart, ystop, 2.5, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 0
    ystop = 2000
    out_img, bboxes9 = find_temoc(out_img, ystart, ystop, 2.6, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                 spatial_size, hist_bins)
    ystart = 0
    ystop = 2000
    out_img, bboxes10 = find_temoc(out_img, ystart, ystop, 1.9, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                  spatial_size, hist_bins)

    bboxes.extend(bboxes1)
    bboxes.extend(bboxes2)
    bboxes.extend(bboxes3)
    bboxes.extend(bboxes4)
    bboxes.extend(bboxes5)
    bboxes.extend(bboxes6)
    bboxes.extend(bboxes7)
    bboxes.extend(bboxes8)
    bboxes.extend(bboxes9)
    bboxes.extend(bboxes10)

    return out_img, bboxes


# def show_images(image1, image2, image3, image1_exp="Image 1", image2_exp="Image 2", image3_exp="Image 3"):
#     f, (ax1, ax2, ax3) = plt.subplots(1, 3)
#     f.tight_layout()
#     ax1.imshow(image1)
#     ax1.set_title(image1_exp, fontsize=20)
#     ax2.imshow(image2)
#     ax2.set_title(image2_exp, fontsize=20)
#     ax3.imshow(image3)
#     ax3.set_title(image3_exp, fontsize=20)
#     plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


if __name__ == '__main__':
    camera = cv2.VideoCapture(0)

    # ------------Explore data---------------- #
    ist = glob.glob('positive_images/*.jpg')
    nott = glob.glob('resizetrain/not/*.jpg')
    print("Size of is-t dataset : ", len(ist))
    print("Size of non-t dataset : ", len(nott))

    # ------------extract features and train SVM---------------- #

    car_features = extract_features(ist, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(nott, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X, y)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC')

    while True:
        (ret, frame) = camera.read()
        if not ret:
            break
        # ------------Find Temoc using sliding window---------------- #
        image = frame
        # image = mpimg.imread('test/11.png')
        draw_image = np.copy(image)
        output_image, bboxes = apply_sliding_window(image, svc, X_scaler, pix_per_cell, cell_per_block, spatial_size,
                                                    hist_bins)

        heat = np.zeros_like(output_image[:, :, 0]).astype(np.float)
        # Add heat to each box in box list
        heat = add_heat(heat, bboxes)

        # Apply threshold to help remove false positives
        threshold = 1
        heat = apply_threshold(heat, threshold)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(image), labels)
        # Output test image result
        cv2.imshow("draw_img", cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
        k = cv2.waitKey(5) & 0xFF
        if k == 5:
            break
    camera.release()
    cv2.destroyAllWindows()