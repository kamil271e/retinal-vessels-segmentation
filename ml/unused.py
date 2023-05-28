# 
# 
#  Set of functions that are no longer in use
# 
# 

import numpy as np
import matplotlib.pyplot as plt
import cv2

def padding(img, n, rgb=True):
    if rgb:
        rows, cols = img[:,:,0].shape
    else:
        rows, cols = img.shape
        
    add_rows, add_cols = n - (rows % n), n - (cols % n)
    append_columns = np.zeros((rows, add_cols), dtype=img.dtype)
    append_rows = np.zeros((add_rows, cols + add_cols), dtype=img.dtype)
    
    if rgb:
        channels = np.zeros((rows+add_rows, cols+add_cols, 3))
        for i in range(3):
            channel = img[:,:,i]
            channel = np.concatenate((channel, append_columns), axis=1)
            channel = np.concatenate((channel, append_rows), axis=0)
            channels[:,:,i] = channel   
        img = np.stack([c for c in channels], axis=0).astype(np.float32)    
    
    else:
        img = np.concatenate((img, append_columns), axis=1)
        img = np.concatenate((img, append_rows), axis=0)
    
    return img / 255.0


def display_sections(img_sections, img_size, n): # test for low resolutions or/and big windows: e.g 100x100, 10x10
    if img_sections.ndim != 3 and img_sections.ndim != 4:
        print('Display available for 2D or 3D sections')
        return
    
    if img_sections.ndim == 3:
        rows, cols = img_size # padded img_size !
    else:
        rows, cols = img_size[0], img_size[1]
        
    rows = int(rows / n)
    cols = int(cols / n)
    
    fig, ax = plt.subplots(rows, cols, figsize=(4,4))
    for i in range(rows):
        for j in range(cols):
            ax[i, j].axis('off')
            ax[i, j].imshow(img_sections[cols * i + j].astype(np.uint8))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()
    
    
def sectioning_(img, target, n, debug=False):
    if img.shape[0] % n != 0 or img.shape[1] % n != 0:
        img = padding(img, n)
        
    if target.shape[0] % n != 0 or target.shape[1] % n != 0:
        target = padding(target, n, rgb=False)
    
    if debug:
        print('Img padded shape: ', img.shape)
        print('Target padded shape: ', target.shape)
    
    rows, cols = int(img.shape[0] / n) , int(img.shape[1] / n)
    num_sections = rows * cols
    img_sections = np.zeros((num_sections, n, n, 3))
    target_sections = np.zeros((num_sections, n, n))

    for i in range(rows):
        for j in range(cols):
            img_sections[i * cols + j] = img[i * n : (i+1) * n, j * n : (j+1) * n]
            target_sections[i * cols + j] = target[i * n : (i+1) * n, j * n : (j+1) * n]
            
    return img_sections, target_sections, img.shape, target.shape

