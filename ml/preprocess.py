import numpy as np
import matplotlib.pyplot as plt
import cv2


def clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_channel_clahe = clahe.apply(l_channel)
    lab_image_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))
    rgb_image_clahe = cv2.cvtColor(lab_image_clahe, cv2.COLOR_LAB2RGB)

    return rgb_image_clahe


def hist_equalization(img):
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    return equalized_img
    
    
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


# we move window by one pixel each time
# if passed image is size MxM this function will return M*M sections of size N*N
# each section correspond to one pixel in the input image
def sectioning_valid(img, n):   
    if n % 2 == 0:
        print('N should be an odd number')
        return
    
    # TODO perform padding similar like in 'sectioning' function    
    rows, cols = img.shape[0], img.shape[1]
    print(rows, cols)
    num_sections = rows * cols
    img_sections = np.zeros((num_sections, n, n, 3))
        
    # border filled by 0s - of size (N-1)
    half_n = (n-1) // 2
    pad_size = ((half_n, half_n), (half_n, half_n), (0, 0))
    img = np.pad(img, pad_size, mode='constant', constant_values=0)

    for i in range(half_n, rows-half_n):
        for j in range(half_n, cols-half_n):
            img_sections[(i - half_n) * cols + j - half_n] = img[i - half_n: i + half_n + 1, j - half_n: j + half_n + 1]
    return img_sections
    
    
# cool but not used in proper solution
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
    
    
# cool but not used in proper solution
def sectioning(img, target, n, debug=False):
    # n x n sections TODO
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
    
 