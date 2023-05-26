import numpy as np
import cv2
from scipy.stats import entropy, skew
from scipy.signal import find_peaks
from numba import njit, prange
from skimage.feature import peak_local_max

# image -> vector of features
def feature_extraction(img, debug=False):
    
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    
    # MEAN INTENISTY
    mean = np.mean(img)

    # STANDARD DEVIATION
    std = np.std(img)
    
    # HISTOGRAM FEATURES
    # using grayscale or only green channel?
    green_channel = img[:,:,1]
    histogram, _ = np.histogram(gray.flatten(), bins=256, range=[0,256])
    peaks, _ = find_peaks(histogram)
    peaks_no = len(peaks)
    entropy_value = entropy(histogram)
    skewness_value = skew(histogram)
    
    # COLOR VARIATIONS
    red_var = np.var(img[:,:,2])
    green_var = np.var(img[:,:,1])
    blue_var = np.var(img[:,:,0])
    
    # HU MOMENTS
    hu_moments = cv2.HuMoments(cv2.moments(gray)).flatten()
        
    if debug:
        print('Mean: ', mean)
        print('Std: ', std)
        print('Peaks No.: ', peaks_no)
        print('Entropy: ', entropy_value)
        print('Skewness: ', skewness_value)
        print('RGB varation: ', [red_var, green_var, blue_var])
        print('HU moments:', hu_moments)
        
    feature_vect = np.array([mean, std, peaks_no, entropy_value, skewness_value, red_var, green_var, blue_var])
    feature_vect = np.concatenate((feature_vect, hu_moments), axis=0)
    
    return feature_vect

# Just an idea to calculate features parallelly
# @njit
# def feature_extraction_(img, debug=False):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # MEAN INTENSITY
#     mean = np.mean(img.astype(np.float64))

#     # STANDARD DEVIATION
#     std = np.std(img.astype(np.float64))

#     # HISTOGRAM FEATURES
#     histogram, _ = np.histogram(gray.flatten(), bins=256, range=[0, 256])
#     peaks = peak_local_max(histogram, min_distance=1, threshold_rel=0.5)
#     peaks_no = len(peaks)
#     entropy_value = entropy(histogram)
#     skewness_value = skew(histogram)

#     # COLOR VARIATIONS
#     red_var = np.var(img[:, :, 2].astype(np.float64))
#     green_var = np.var(img[:, :, 1].astype(np.float64))
#     blue_var = np.var(img[:, :, 0].astype(np.float64))

#     # HU MOMENTS
#     moments = cv2.moments(gray)
#     hu_moments = cv2.HuMoments(moments).flatten()

#     if debug:
#         print('Mean: ', mean)
#         print('Std: ', std)
#         print('Peaks No.: ', peaks_no)
#         print('Entropy: ', entropy_value)
#         print('Skewness: ', skewness_value)
#         print('RGB variation: ', [red_var, green_var, blue_var])
#         print('HU moments:', hu_moments)

#     feature_vect = np.array([mean, std, peaks_no, entropy_value, skewness_value, red_var, green_var, blue_var])
#     feature_vect = np.concatenate((feature_vect, hu_moments), axis=0)

#     return feature_vect