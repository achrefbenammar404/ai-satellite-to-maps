import os
from os import listdir
from numpy import asarray
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def load_images(path, size=(256,512)):
    """
    Load and split images into satellite and map pairs.
    Expects directory of images where each image is composed of
    [satellite|map] side by side.
    """
    src_list, tar_list = list(), list()
    for filename in listdir(path):
        # skip non-image files if any
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        pixels = load_img(os.path.join(path, filename), target_size=size)
        pixels = img_to_array(pixels)
        # split into satellite and map
        sat_img, map_img = pixels[:, :256], pixels[:, 256:]
        src_list.append(sat_img)
        tar_list.append(map_img)
    return [asarray(src_list), asarray(tar_list)]

def preprocess_data(data):
    # scale from [0,255] to [-1,1]
    X1, X2 = data[0], data[1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]
