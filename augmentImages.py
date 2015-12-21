#Preprocess training images.
#Scale 300 seems to be sufficient; 500 and 1000 are overkill
import cv2, glob, numpy as np
from joblib import Parallel, delayed
from utils import bbox, scaleRadius, random_crops, get_time
from utils import get_distorted_img, files_list, unsharp_img
import sys
import os
import caffe

root_folder = '/home/ubuntu/dataset/'
in_folder_train = 'train/'
in_folder_val = 'validation/'
in_folder_test = 'test/test/'
pb = [0.005, 0.80, 0.35, 1., 1.]


def process_img(fname_label, crop_shape, scale, random_draws, mode, logging=True):
    imgs = []
    name, label = fname_label
    if logging:
        print "%s [%d] Processing file %s" % (get_time(), os.getpid(), name)
    a = (caffe.io.load_image(name) * 255).astype(np.uint8)
    a = scaleRadius(a,scale)
    if a is None:
        print "%s [%d] Unable to retrieve scaleRadius() img for file %s" % (get_time(), os.getpid(), name)
        for i in range(random_draws + 1):
            imgs.append(np.zeros((crop_shape[0], crop_shape[1], 3), dtype=np.uint8))
        return imgs

    ua = unsharp_img(a, scale)
    ca = random_crops(ua, shape=crop_shape)
    imgs.append(ca)
    
    # Don't augment validation set
    if 'val' in mode:
        return imgs
    
    # Check if augmentation is needed
    if mode is "train" and np.random.uniform(0, 1) > pb[label]:
        return imgs
    
    for i in range(random_draws):
        dist_img = get_distorted_img(ua, mode)
        out_im = random_crops(dist_img, shape=crop_shape)
        imgs.append(out_im)

    return imgs

def write_imgs(imgs, name, crop_shape, mode):
    if imgs is None:
        print "%s [%d] WARN: File not processed: %s" % (get_time(), os.getpid(), name)
        return
    new_name = ''
    if "train" in mode:
        new_name = name.replace("train/", "%d_train_aug/" % crop_shape[0])
    elif "val" in mode:
        new_name = name.replace("validation/", "%d_val/" % crop_shape[0])
    else:
        return
    for i, im in enumerate(imgs):
        if i == 0:
            cv2.imwrite(new_name, im)
            continue
        out_name = new_name.replace(".jpeg", "_%d.jpeg" % (i))
        out_im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_name, out_im)

def main_proc(fname_label, crop_shape, scale, random_draws, mode="train"):
    ferr = open("out_%d.log" % os.getpid(), 'a')
    sys.stdout = ferr
    sys.stderr = ferr
    imgs = process_img(fname_label, crop_shape, scale, random_draws, mode)
    write_imgs(imgs, fname_label[0], crop_shape, mode)
    ferr.close()

def main(scale, crop_shape, random_draws, mode):
    in_fold = root_folder + in_folder_train
    if "val" in mode:
        in_fold = root_folder + in_folder_val
    elif "test" in mode:
        in_fold = root_folder + in_folder_test

    print "Reading images from %s" % in_fold
    print "Run mode: %s" % mode.upper()
    names = files_list(in_fold, mode)
    print "Total number of files: %d" % len(names)
    
    # Create a parallel pool
    errf = open('err.log', 'w')
    sys.stderr = errf
    with Parallel(n_jobs=8) as parallel:
        rets = parallel(delayed(main_proc)(fname_label, crop_shape, scale, random_draws, mode) 
                        for fname_label in names)
        print "Done. A total of %d files processed." % len(rets)
    errf.close()

    
if __name__ == '__main__':
    scale = 180
    crop_shape = (256, 256)
    random_draws = 7
    mode = "train"  # "val", "test", "train"
    main(scale, crop_shape, random_draws, mode)
