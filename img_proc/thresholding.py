import cv2
import numpy as np

# CLAHE - Contrast Limited Adaptive Histogram Equalization
def CLAHE(img):
    # extract green channel
    img_green = img[:,:,1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img_green)
    return clahe_img


def median_filtering(img, ksize=3):
    filtered_img = cv2.medianBlur(img, ksize=ksize)
    return filtered_img


# Steps of thresholding
# 1) Initialize a mean filter of window size N x N 
# 2) Convolve previously enhanced image with the mean filter
# 3) Calculate difference by substracting convolved image from initial one
# 4) Threshold the difference image with C value
# 5) Calculate complement of obtained image

def mean_C_thresholding(img, n=13, C=-15):
    meanfilter = np.ones((n,n)) / n**2
    meanfiltered = cv2.filter2D(img, -1, meanfilter)
    diff = np.double(img) - meanfiltered 
    thresholded = cv2.threshold(diff, C, 255, cv2.THRESH_BINARY)[1] 
    complemented = cv2.bitwise_not(thresholded) 
    return complemented


def postprocess(segmented, mask):
    segmented[np.isnan(segmented)] = 0
    segmented = (segmented - segmented.min()) / (segmented.max() - segmented.min())
    segmented *= 255
    segmented_final = cv2.bitwise_and(mask.astype('float64'), segmented)
    return segmented_final


def score(segmented, target):
    segmented[segmented > 0] = 1
    target[target > 0] = 1
    segmented = segmented.astype('uint8')
    predicted = 1 - cv2.bitwise_xor(segmented, target) # XNOR 0,0 -> 1; 1,1 -> 1
    accuracy = np.sum(predicted) / (predicted.shape[0] * predicted.shape[1])
    return accuracy