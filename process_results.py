import os

import numpy as np
from keras_preprocessing.image import save_img


def labelVisualize(num_class, color_dict, img):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255


def saveResult(save_path, npyfile):
    for i, item in enumerate(npyfile):
        #img = item[:, :, 1]
        save_img(os.path.join(save_path, "%d_predict.png" % i), item)