
# 
# 
#  Set of functions that are no longer in use
# 
# 

import numpy as np
import cv2


def sharpening(img):
    kernel = np.array([[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]])
    img_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    return img_sharp


def hist_equalization(img):
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    return equalized_img


def edge_detection(img, lvl=25):
    edges = cv2.Canny(img, lvl, lvl)
    return edges


def denoising(img, convert=False):
    if convert:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    return denoised


