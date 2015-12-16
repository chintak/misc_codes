#Preprocess training images.
#Scale 300 seems to be sufficient; 500 and 1000 are overkill
import cv2, glob, numpy as np
from joblib import Parallel, delayed
from utils import bbox, scaleRadius, random_crops, get_time, flushfile
import sys
import os

scale = 200
crop_shape = (304, 304)
root_folder = '/home/ubuntu/dataset/'
in_folder_train = 'train/'
in_folder_val = 'validation/'


def process_img(name, crop_shape, scale):
    ferr = open("out_%d.log" % os.getpid(), 'a')
    sys.stdout = ferr
    sys.stderr = ferr
    print "%s [%d] Processing file %s" % (get_time(), os.getpid(), name)
    a = cv2.imread(name)
    a = bbox(scaleRadius(a,scale))
    if a is None:
        ferr.close()
        return
    b = np.zeros(a.shape)
    cv2.circle(b,(a.shape[1]//2,a.shape[0]//2),int(scale*0.9),(1,1,1),-1,8,0)
    aa = cv2.addWeighted(a,4,cv2.GaussianBlur(a,(0,0),scale/30),-4,128)*b+128*(1-b)
    rand_im = random_crops(aa, shape=crop_shape)
    if "train/" in name:
        new_name = name.replace("train/", "%d_train/" % scale)
    elif "validation/" in name:
        new_name = name.replace("validation/", "%d_val/" % scale)
    cv2.imwrite(new_name,rand_im)
    ferr.close()

def main():
    names = []
    for r, ds, fs in os.walk(root_folder + in_folder_train):
        for f in fs:
            if '.jpeg' not in f:
                continue
            names.append(os.path.join(r, f))
    for r, ds, fs in os.walk(root_folder + in_folder_val):
        for f in fs:
            if '.jpeg' not in f:
                continue
            names.append(os.path.join(r, f))
    # Create a parallel pool
    errf = open('err.log', 'w')
    sys.stderr = errf
    with Parallel(n_jobs=8) as parallel:
        print len(names)
        rets = parallel(delayed(process_img)(fname, crop_shape, scale) 
                        for j, fname in enumerate(names))
        print "Done. A total of %d files processed." % len(rets)
    errf.close()

    
if __name__ == '__main__':
    main()
