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


# we move window by one pixel each time
# if passed image is size MxM this function will return M*M sections of size N*N
# each section correspond to one pixel in the input image
def sectioning_img(img, n):
    if n % 2 == 0:
        print("N should be an odd number")
        return

    img = clahe(img)
    img = hist_equalization(img)

    # TODO perform padding similar like in unused 'sectioning' function
    rows, cols = img.shape[0], img.shape[1]
    num_sections = rows * cols
    img_sections = np.zeros((num_sections, n, n, 3))

    # border filled by 0s - of size (N-1)
    half_n = (n - 1) // 2
    pad_size = ((half_n, half_n), (half_n, half_n), (0, 0))
    img = np.pad(img, pad_size, mode="constant", constant_values=0)

    for i in range(half_n, rows - half_n):
        for j in range(half_n, cols - half_n):
            img_sections[(i - half_n) * cols + j - half_n] = img[
                i - half_n : i + half_n + 1, j - half_n : j + half_n + 1
            ]
    return img_sections


# sectioning num_img nuber of randomly sampled images
def sectioning(X, y, num_img, img_size, n, indices=[]):
    sections = np.zeros((num_img, img_size[0] * img_size[1], n, n, 3))
    targets = np.zeros((num_img, img_size[0] * img_size[1]))

    if indices:
        imgs_indx = indices
        if num_img != len(indices):
            print("Number of indices are not equal number of images")
            return
    else:
        imgs_indx = [np.random.randint(0, len(X) - 1) for _ in range(num_img)]

    for i, indx in enumerate(imgs_indx):
        sections[i] = sectioning_img(X[indx], n)
        _, target = cv2.threshold(y[indx], 128, 1, cv2.THRESH_BINARY)
        targets[i] = target.flatten()

    sections = sections.reshape(num_img * img_size[0] * img_size[1], n, n, 3)
    targets = targets.flatten()

    return sections, targets
