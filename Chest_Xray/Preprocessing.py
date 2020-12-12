import numpy as np
import os
from keras.preprocessing.image import image


def preprocessing(path, X, y):
    img_list = os.listdir(path)
    for img_name in img_list:
        img_path = path + '/' + img_name
        img = image.load_img(img_path, color_mode="grayscale", target_size=(150, 150))
        img_tensor = image.img_to_array(img)
        X.append(img_tensor)
        if path.split("/")[-1] == "NORMAL":
            y.append([0])
        else:
            y.append([1])


def save_data(X_train, X_val, X_test, y_train, y_val, y_test):
    np.save("D:/dataset/train/X_train.npy", X_train)
    np.save("D:/dataset/val/X_val.npy", X_val)
    np.save("D:/dataset/test/X_test.npy", X_test)
    np.save("D:/dataset/train/y_train.npy", y_train)
    np.save("D:/dataset/val/y_val.npy", y_val)
    np.save("D:/dataset/test/y_test.npy", y_test)


train_normal_path = "D:/X-Ray image/train/NORMAL"
train_pneumonia_path = "D:/X-Ray image/train/PNEUMONIA"
val_normal_path = "D:/X-Ray image/val/NORMAL"
val_pneumonia_path = "D:/X-Ray image/val/PNEUMONIA"
test_normal_path = "D:/X-Ray image/test/NORMAL"
test_pneumonia_path = "D:/X-Ray image/test/PNEUMONIA"

X_train, y_train = [], []
X_val, y_val = [], []
X_test, y_test = [], []

preprocessing(train_normal_path, X_train, y_train)
preprocessing(train_pneumonia_path, X_train, y_train)
X_train = np.array(X_train)
y_train = np.array(y_train)

preprocessing(val_normal_path, X_val, y_val)
preprocessing(val_pneumonia_path, X_val, y_val)
X_val = np.array(X_val)
y_val = np.array(y_val)

preprocessing(test_normal_path, X_test, y_test)
preprocessing(test_pneumonia_path, X_test, y_test)
X_test = np.array(X_test)
y_test = np.array(y_test)

X_train_mean = np.mean(X_train)
X_train_std = np.std(X_train)
X_train = (X_train - X_train_mean) / X_train_std
X_val = (X_val - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std

save_data(X_train, X_val, X_test, y_train, y_val, y_test)