import os
import cv2
import requests
import zipfile
import shutil

MAX_HEIGHT, MAX_WIDTH = 3504, 2336
MIN_HEIGHT, MIN_WIDTH = 10, 10
URL = "https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip"


def load(img_size):
    img_size = (img_size[1], img_size[0])

    if (img_size[0] > MAX_HEIGHT or img_size[0] < MIN_HEIGHT) or (
        img_size[1] > MAX_WIDTH or img_size[1] < MIN_WIDTH
    ):
        print("Incorrect image resolution")
        return None

    images_dir = "../images"
    targets_dir = "../manual1"
    masks_dir = "../mask"

    if (
        not os.path.exists(images_dir)
        or not os.path.exists(targets_dir)
        or not os.path.exists(masks_dir)
    ):
        download_data()

    X, y, z = [], [], []

    images = sorted(os.listdir(images_dir))
    targets = sorted(os.listdir(targets_dir))
    masks = sorted(os.listdir(masks_dir))

    for img_dir, target_dir, mask_dir in zip(images, targets, masks):
        img = cv2.imread(os.path.join(images_dir, img_dir))
        img = cv2.resize(img, img_size)
        target = cv2.imread(os.path.join(targets_dir, target_dir), cv2.IMREAD_GRAYSCALE)
        target = cv2.resize(target, img_size)
        mask = cv2.imread(os.path.join(masks_dir, mask_dir), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size)
        mask[mask > 0] = 255.0
        X.append(img)
        y.append(target)
        z.append(mask)

    return X, y, z


def download_data():
    print("Downloading and extracting data...")
    download_path = "./data.zip"

    response = requests.get(URL)
    with open(download_path, "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall("../")

    os.remove(download_path)
