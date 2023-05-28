import numpy as np
import cv2

def train_val_test_split(X, y, train_size = 2/3):
    indices = np.random.permutation(len(X))
    
    train_samples = int(train_size * len(X))
    val_samples = int((1 - train_size) * len(X) / 1.5)
    
    train_indices = indices[:train_samples]
    val_indices = indices[train_samples:train_samples+val_samples]
    test_indices = indices[train_samples+val_samples:]
    
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