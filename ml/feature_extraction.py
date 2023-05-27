import numpy as np
import cv2

num_features = 13

def extract_features(img):
    means, stds = cv2.meanStdDev(img)
    stats = np.concatenate([means, stds], axis=0).flatten()
    grayscale_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    hu_moments = cv2.HuMoments(cv2.moments(grayscale_img)).flatten()
    feature_vect = np.concatenate((stats, hu_moments), axis=0)
    return feature_vect

def convert_to_feature_vec(sections):
    section_features = np.zeros((sections.shape[0], num_features))

    for i in range(sections.shape[0]):
        section_features[i] = extract_features(sections[i])
    
    return section_features