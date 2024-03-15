import numpy as np
import cv2


def clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_channel_clahe = clahe.apply(l_channel)
    lab_image_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))
    rgb_image_clahe = cv2.cvtColor(lab_image_clahe, cv2.COLOR_LAB2RGB)

    return rgb_image_clahe


def train_val_test_split(X, y, train_size=2 / 3):
    np.random.seed(13)
    indices = np.random.permutation(len(X))

    train_samples = int(train_size * len(X))
    val_samples = int((1 - train_size) * len(X) / 2)

    train_indices = indices[:train_samples]
    val_indices = indices[train_samples : train_samples + val_samples]
    test_indices = indices[train_samples + val_samples :]

    print(len(train_indices), len(val_indices), len(test_indices))

    X = np.array(X) / 255

    y = [cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)[1] for img in y]
    y = np.array(y)
    y = np.expand_dims(y, -1)

    X_train = X[train_indices]
    X_val = X[val_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_val = y[val_indices]
    y_test = y[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test
