#!/usr/bin/python

import sys
import os
import numpy as np
import cv2
from joblib import Parallel, delayed
from data_augmentation import data_augmentation
from utils import bbox, create_fixed_image_shape


temp_shape = (350, 350, 3)
out_shape = (256, 256, 3)
root_folder = '/home/ubuntu/dataset/'
in_folder = 'validation_bkp/'
out_folder = 'val_all_aug/'
fsock = open('out.log', 'w')
# sys.stdout = fsock
sys.stderr = fsock


def proc(name, k):
    im = cv2.imread(name)
    imre = create_fixed_image_shape(bbox(im), temp_shape,
                                    random_fill=False, mode='fit')
    imgs = data_augmentation(imre, frame_size=out_shape)
    for i in range(len(imgs)):
        img = imgs[i]
        imgs.append(img[:, ::-1, :])
    for i, img in enumerate(imgs):
        new_name = name.replace(in_folder, out_folder)
        new_name = new_name.replace('.jpeg', '_%d.jpeg' % i)
        cv2.imwrite(new_name, img)
    if k % 10 == 0:
        print "Completed %d." % k


names = []
for r, ds, fs in os.walk(root_folder + in_folder):
    for f in fs:
        if '.jpeg' not in f:
            continue
        names.append(os.path.join(r, f))

with Parallel(n_jobs=8) as parallel:
    parallel(delayed(proc)(fname, k) for k, fname in enumerate(names))

fsock.close()
