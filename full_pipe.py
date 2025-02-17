import os
import random
import sys
import cv2
import numpy as np
import xgboost as xgb
import segmentation_models as sm

os.environ["SM_FRAMEWORK"] = "tf.keras"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from load_images import load

from img_proc.thresholding import (
    CLAHE,
    median_filtering,
    mean_C_thresholding,
    postprocess,
    score,
)

from ml.preprocess import clahe, sectioning, sectioning_img, hist_equalization
from ml.feature_extraction import extract_features, convert_to_feature_vec

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    Callback,
    ModelCheckpoint,
    EarlyStopping,
)
from tensorflow.keras.optimizers import Adam
from dl.preprocess import train_val_test_split

IMG_SIZE = (512, 512)


def img_proc_approach(N=5):
    X, y, masks = load(IMG_SIZE)

    # N random photos
    accumulated_targets = []
    accumulated_segmentations = []
    random_imgs_indx = [random.randint(0, len(X) - 1) for _ in range(N)]

    for i in random_imgs_indx:
        clahed = CLAHE(X[i])
        m_filtered = median_filtering(clahed, ksize=1)
        segmented = mean_C_thresholding(m_filtered)
        segmented_final = postprocess(segmented, masks[i])

        accumulated_targets.append(y[i])
        accumulated_segmentations.append(segmented_final)

    _ = score(accumulated_segmentations, accumulated_targets)
    print("IMG_PROC-DONE")


def ml_approach(N=5, num_img=15, section_size=5, num_features=13):
    # TODO refactor this
    X, y, masks = load(IMG_SIZE)
    X_unseen, y_unseen = X[:N], y[:N]
    X, y = X[N:], y[N:]

    sections, targets = sectioning(X, y, num_img, IMG_SIZE, N)
    section_features = convert_to_feature_vec(sections)
    X_train, X_test, y_train, y_test = train_test_split(
        section_features, targets, test_size=0.2, shuffle=True
    )
    xgb_clf = xgb.XGBClassifier()
    xgb_clf.fit(X_train, y_train)

    # Evaluate
    y_hat = xgb_clf.predict(X_test)
    j = np.random.randint(0, N)
    sections = sectioning_img(X_unseen[j], section_size)
    X_new = np.zeros((sections.shape[0], num_features))

    for i in range(sections.shape[0]):
        X_new[i] = extract_features(sections[i])

    y_new = y_unseen[j].flatten()
    _, y_new = cv2.threshold(y_new, 128, 1, cv2.THRESH_BINARY)

    _ = xgb_clf.predict(X_train)
    _ = xgb_clf.predict(X_test)
    _ = xgb_clf.predict(X_new)
    print("ML-DONE")


def dl_approach(
    batch_size=2, epochs=40, backbone="efficientnetb0", lr=0.0001, patience=12
):
    X, y, _ = load(IMG_SIZE)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, train_size=7 / 9
    )

    train_datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

    # Training
    model = sm.Unet(backbone, classes=1, activation="sigmoid")
    model.compile(optimizer=Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])

    _ = model.fit(
        train_generator,
        validation_data=val_generator,
        callbacks=[
            ModelCheckpoint(f"unet_{backbone}.keras", save_best_only=True),
            EarlyStopping(patience=patience),
        ],
        epochs=epochs,
    )

    y_hat = model.predict(X_test)
    y_hat = y_hat.flatten()
    y_test = y_test.flatten()
    _, y_hat = cv2.threshold(y_hat, 0.5, 1, cv2.THRESH_BINARY)
    y_hat = y_hat.astype(np.uint8)
    _ = classification_report(y_hat, y_test)
    print("DL-DONE")


if __name__ == "__main__":
    img_proc_approach(N=1)
    ml_approach(N=1, num_img=5)
    dl_approach(epochs=1)
