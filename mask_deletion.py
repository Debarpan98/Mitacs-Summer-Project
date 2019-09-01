# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:34:54 2019

@author: dell
"""

import os
import numpy
import random
from keras.preprocessing.image import load_img, img_to_array

blank_list = []
#masks_path = '/home/debarpan/projects/def-erangauk-ab/debarpan/uog_project/data/masks-new/'
#images_path= '/home/debarpan/projects/def-erangauk-ab/debarpan/uog_project/data/images-new/'
masks_path = 'masks-112/'
images_path='images-112/'
for mask in os.listdir(masks_path):
    mask = load_img(masks_path + mask)
    np_mask = img_to_array(mask)
    max_array = numpy.amax(np_mask)
    if max_array == 0:
        blank_list.append(np_mask)

l = len(blank_list)
l=(int)(0.80*l)
c=0
mask_names = sorted(next(os.walk(masks_path))[2])
image_names = sorted(next(os.walk(images_path))[2])
zipped_list = list(zip(image_names, mask_names))
random.shuffle(zipped_list)
image_names , mask_names = zip(*zipped_list)
for i in range(len(mask_names)):
    if c<=l:
        mask = load_img(masks_path + mask_names[i])
        np_mask = img_to_array(mask)
        max_array = numpy.amax(np_mask)
        if max_array == 0:
            c=c+1
            os.remove(images_path + image_names[i])
            os.remove(masks_path + mask_names[i])
            

#for filename in os.listdir(images_path): 
#        src = 'images-112/' + filename 
#        dst = 'images-112/' + filename[:-5] + ".png"
#        os.rename(src, dst) 

#for i in range(len(mask_names)):
#    if mask_names[i] != image_names[i]:
#        print(0)
#        break
#    