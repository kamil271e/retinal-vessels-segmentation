import numpy as np
import cv2

# image section -> vector of features
def extract_features(img):
    means, stds = cv2.meanStdDev(img)
    stats = np.concatenate([means, stds], axis=0).flatten()
    grayscale_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    hu_moments = cv2.HuMoments(cv2.moments(grayscale_img)).flatten()
    feature_vect = np.concatenate((stats, hu_moments), axis=0)
    return feature_vect

