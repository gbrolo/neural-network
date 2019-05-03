# Utils for loading images
# @author: gbrolo

import os
import cv2
import numpy as np

# Returns array of training images and array of labels for each image
# class_names: list of strings
# class_ids: dictionary with key: class, value: id for class
# data_path: path to images (data directory)
# train_number: integer with how many training images will you use for each category
def load_train_images(class_names, class_ids, data_path, train_number):
    print('Loading images...')
    X = []
    y = []
    
    for classname in class_names:
        class_folder = os.path.join(data_path, classname)
        images = os.listdir(class_folder)[0:train_number]

        for image in images:
            img = np.ravel(np.array(np.float32(cv2.imread(os.path.join(class_folder, image), 0))))
            X.append(img)
            y.append(class_ids[classname])

    print('Loaded images!')
    return np.array(X), np.array(y)

def shuffle_dataset(X, y):
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X[p], y[p]

def partition_dataset(X, y, train_number):
    X_train = X[0 : int((train_number * 0.80))]
    X_cross = X[int((train_number * 0.80)) : int(((train_number * 0.80) + (train_number * 0.10)))]
    X_test = X[int(((train_number * 0.80) + (train_number * 0.10))) : int((train_number))]

    y_train = y[0 : int((train_number * 0.80))]
    y_cross = y[int((train_number * 0.80)) : int(((train_number * 0.80) + (train_number * 0.10)))]
    y_test = y[int(((train_number * 0.80) + (train_number * 0.10))) : int((train_number))]

    return X_train, X_cross, X_test, y_train, y_cross, y_test

# loads test image from client, for server usage in classifying
def load_test_image():    
    X = []
    class_folder = 'drawings'
    images = os.listdir(class_folder)

    for image in images:
        img = np.ravel(np.array(np.float32(cv2.imread(os.path.join(class_folder, image), 0))))
        X.append(img)
            
    return np.array(X)