import os
import json
import torch
import cv2 as cv

f = open(r"E:/codes/dehaze_cycle_gan/configs.json")
configs = json.load(f)

height = configs["img_specifications"]["height"]
width = configs["img_specifications"]["width"]


def rotate_to_landscape(path):
    for file_path in os.listdir(path):
        img_path = os.path.join(path, file_path)
        img = cv.imread(img_path)
        if img.shape[1] < img.shape[0]:
            img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
            cv.imwrite(img_path, img)


def downscale_images(path):
    for file_path in os.listdir(path):
        img_path = os.path.join(path, file_path)
        img = cv.imread(img_path)
        img = cv.resize(img, (width, height), interpolation=cv.INTER_AREA)
        cv.imwrite(img_path, img)


x_training_dataset_path = configs["path"]["dataset_path"]
x_training_dataset_path = os.path.join(x_training_dataset_path, "Train/X")
y_training_dataset_path = configs["path"]["dataset_path"]
y_training_dataset_path = os.path.join(y_training_dataset_path, "Train/Y")

x_validation_dataset_path = configs["path"]["dataset_path"]
x_validation_dataset_path = os.path.join(x_validation_dataset_path, "Validation/X")
y_validation_dataset_path = configs["path"]["dataset_path"]
y_validation_dataset_path = os.path.join(y_validation_dataset_path, "Validation/Y")

rotate_to_landscape(r"E:/codes/dehaze_cycle_gan/dataset/Train/X")
downscale_images(r"E:/codes/dehaze_cycle_gan/dataset/Train/X")

rotate_to_landscape(r"E:/codes/dehaze_cycle_gan/dataset/Train/Y")
downscale_images(r"E:/codes/dehaze_cycle_gan/dataset/Train/Y")

rotate_to_landscape(r"E:/codes/dehaze_cycle_gan/dataset/Validation/X")
downscale_images(r"E:/codes/dehaze_cycle_gan/dataset/Validation/X")

rotate_to_landscape(r"E:/codes/dehaze_cycle_gan/dataset/Validation/Y")
downscale_images(r"E:/codes/dehaze_cycle_gan/dataset/Validation/Y")


def load_training_images(index, batch_size, normalised=True):
    x_img_path = os.listdir(x_training_dataset_path)[0]
    x_img_path = os.path.join(x_training_dataset_path, x_img_path)
    x_img = cv.imread(x_img_path)

    y_img_path = os.listdir(y_training_dataset_path)[0]
    y_img_path = os.path.join(y_training_dataset_path, y_img_path)
    y_img = cv.imread(y_img_path)

    X = torch.tensor(x_img).reshape((1, 3, width, height))
    Y = torch.tensor(y_img).reshape((1, 3, width, height))

    for img_path in os.listdir(x_training_dataset_path)[index + 1 : index + batch_size]:
        x_img_path = os.path.join(x_training_dataset_path, img_path)
        x_img = cv.imread(x_img_path)
        X = torch.tensor(X, dtype=torch.float)
        x_img = torch.tensor(x_img, dtype=torch.float).reshape((1, 3, width, height))
        X = torch.cat((X, x_img), 0)

    for img_path in os.listdir(y_training_dataset_path)[index + 1 : index + batch_size]:
        y_img_path = os.path.join(y_training_dataset_path, img_path)
        y_img = cv.imread(y_img_path)
        Y = torch.tensor(Y, dtype=torch.float)
        y_img = torch.tensor(y_img, dtype=torch.float).reshape((1, 3, width, height))
        Y = torch.cat((Y, y_img), 0)

    if normalised:
        X = X * (1 / 127.5) - 1
        Y = Y * (1 / 127.5) - 1

    return X, Y


def load_validation_images(index, normalised=True):
    x_img_path = os.listdir(x_validation_dataset_path)[index]
    x_img_path = os.path.join(x_validation_dataset_path, x_img_path)
    x_img = cv.imread(x_img_path)

    y_img_path = os.listdir(y_validation_dataset_path)[index]
    y_img_path = os.path.join(y_validation_dataset_path, y_img_path)
    y_img = cv.imread(y_img_path)

    X = torch.tensor(x_img).reshape((1, 3, width, height))
    Y = torch.tensor(y_img).reshape((1, 3, width, height))

    if normalised:
        X = X * (1 / 127.5) - 1
        Y = Y * (1 / 127.5) - 1

    return X, Y


def get_dataset_info():
    return {
        "number_of_training_images": len(os.listdir(x_training_dataset_path)),
        "number_of_validation_images": len(os.listdir(x_validation_dataset_path)),
    }
