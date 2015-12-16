#!/usr/bin/python

import sys
import os
import numpy as np
import cv2
from joblib import Parallel, delayed
from utils import bbox, create_fixed_image_shape


temp_shape = (350, 350, 3)
out_shape = (256, 256, 3)
root_folder = '/home/ubuntu/dataset/'
in_folder = 'validation_bkp/'
out_folder = 'val_all_aug/'
fsock = open('out.log', 'w')
# sys.stdout = fsock
sys.stderr = fsock


def proc_fit(name, k):
    im = cv2.imread(name)
    img = create_fixed_image_shape(bbox(im), out_shape,
                                   random_fill=False, mode='fit')
    new_name = name.replace(in_folder, out_folder)
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
    parallel(delayed(proc_fit)(fname, k) for k, fname in enumerate(names))

fsock.close()
