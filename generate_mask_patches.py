#41
import shutil

import skimage.io as io
import numpy
from skimage.measure import block_reduce
import os.path

import imageio

downsample_factor = 4
patch_dim = 256
path_src = '/home/fabian/projects/def-erangauk-ab/fabian/liver-data/py_wsi/data/masks/'
path_dst = '/home/fabian/projects/def-erangauk-ab/fabian/liver-data/py_wsi/db/masks/'
#path_src = 'data/py_wsi/data/masks/'
#path_dst = 'data/py_wsi/db/masks/'

shutil.rmtree(path_dst)
os.mkdir(path_dst)

_, _, files = next(os.walk(path_src))
for idx,file in enumerate(files):
    image_name = file[:10]
    img = io.imread("{}{}".format(path_src, file))
    img_resize = block_reduce(img, block_size=(downsample_factor, downsample_factor), func=numpy.max)
    ID = 0
    for idx_j, j in enumerate(range(0, img_resize.shape[1], patch_dim)):
        if j + patch_dim > img_resize.shape[1]:
            break
        for idx_i,i in enumerate(range(0, img_resize.shape[0], patch_dim)):
            if i + patch_dim > img_resize.shape[0]:
                break

            mask_patch = img_resize[i:(patch_dim + i), j:(patch_dim + j)] * 255
            a, b = mask_patch.shape
            if a == patch_dim and b == patch_dim:
                imageio.imwrite("{}{}_{}_{}.png".format(path_dst, image_name, idx_j, idx_i), mask_patch)
                ID += 1

exit()
