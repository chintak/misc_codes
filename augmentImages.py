#Preprocess training images.
#Scale 300 seems to be sufficient; 500 and 1000 are overkill
import cv2, glob, numpy as np
from joblib import Parallel, delayed
from utils import bbox, scaleRadius, random_crops, get_time
from utils import get_distorted_img
import sys
import os

scale = 200
crop_shape = (304, 304)
root_folder = '/home/ubuntu/dataset/'
in_folder_train = 'train/'
in_folder_val = 'validation/'
random_draws = 4
pb = [0.01, 0.80, 0.35, 1., 1.]


def unsharp_img(a):
    b = np.zeros(a.shape)
    cv2.circle(b,(a.shape[1]//2,a.shape[0]//2),int(scale*0.9),(1,1,1),-1,8,0)
    aa = cv2.addWeighted(a,4,cv2.GaussianBlur(a,(0,0),scale/30),-4,128)*b+128*(1-b)
    aa = aa.astype(np.uint8)
    return aa


def process_img(fname_label, crop_shape, scale):
    name, label = fname_label
    ferr = open("out_%d.log" % os.getpid(), 'a')
    sys.stdout = ferr
    sys.stderr = ferr
    print "%s [%d] Processing file %s" % (get_time(), os.getpid(), name)
    a = cv2.imread(name)
    a = scaleRadius(a,scale)
    if a is None:
        ferr.close()
        return

    if "train/" in name:
        new_name = name.replace("train/", "%d_train_aug/" % scale)
    elif "validation/" in name:
        new_name = name.replace("validation/", "%d_val_aug/" % scale)
    ua = unsharp_img(a)
    ca = random_crops(ua, shape=crop_shape)
    cv2.imwrite(new_name, ca)
    
    # Check if augmentation is needed
    if np.random.uniform(0, 1) > pb[label]:
        return
    for i in range(random_draws):
        dist_img = get_distorted_img(ua)
        out_name = new_name.replace(".jpeg", "_%d.jpeg" % (i))
        out_im = random_crops(dist_img, shape=crop_shape)
        cv2.imwrite(out_name, out_im)

    ferr.close()

def main():
    names = []
    for r, ds, fs in os.walk(root_folder + in_folder_train):
        for f in fs:
            if '.jpeg' not in f:
                continue
            label = int(r.strip('/').split('/')[-1])
            if label != 2:
                continue
            names.append((os.path.join(r, f), label))
    for r, ds, fs in os.walk(root_folder + in_folder_val):
        for f in fs:
            if '.jpeg' not in f:
                continue
            names.append((os.path.join(r, f), int(r.strip('/').split('/')[-1])))
    # Create a parallel pool
    errf = open('err.log', 'w')
    sys.stderr = errf
    with Parallel(n_jobs=8) as parallel:
        print len(names)
        rets = parallel(delayed(process_img)(fname_label, crop_shape, scale) 
                        for fname_label in names)
        print "Done. A total of %d files processed." % len(rets)
    errf.close()

    
if __name__ == '__main__':
    main()
