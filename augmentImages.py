# Preprocess training images.
# Scale 300 seems to be sufficient; 500 and 1000 are overkill
import cv2
import glob
import numpy as np
import pandas as pd
from pandas import read_csv
from joblib import Parallel, delayed
from utils import bbox, random_crops, scale_radius, pad_img
from utils import get_distorted_img, unsharp_img
from utils import parse_folder, image_load, make_folder_tree, get_time
from utils import extract_filename_in_path, subsample_inner_circle_img
import sys
import os
import caffe

pb = [0, 0.8, 0.2, 1., 1.]


def sample_train_val_split(names, pdlab, valsplit):
    pdlab = pdlab.reset_index()
    counts = pd.Series(pdlab.label).value_counts()
    samples_per_label = {}
    total_num = pdlab.count()['label']
    val_idx = np.zeros(total_num, dtype=np.uint8)
    rng = np.random.RandomState(1234)
    for k in counts.keys():
        samples_per_label[k] = valsplit // len(counts.keys())
        b = np.arange(counts[k])
        rng.shuffle(b)
        b = b[:samples_per_label[k]]
        val_idx[pdlab.index[pdlab.label == k][b]] = 1
    pdlab['val'] = pd.Series(val_idx, index=pdlab.index)
    pdlab = pdlab.set_index('image')
    return pdlab

def center_crop(img, shape):
    if img.shape == shape:
        return img
    center = img.shape[0] // 2, img.shape[1] // 2
    if img.shape[0] > shape[0] and img.shape[1] > shape[1]:
        im = img[center[0] - shape[0] // 2:center[0] + shape[0] // 2,
                 center[1] - shape[1] // 2:center[1] + shape[1] // 2,
                 :]
    elif img.shape[0] <= shape[0]:
        im = img[:,
                 center[1] - shape[1] // 2:center[1] + shape[1] // 2,
                 :]
    else:
        im = img[center[0] - shape[0] // 2:center[0] + shape[0] // 2,
                 :,
                 :]
    return im[:shape[0], :shape[1], :]

def process_img(name, label, crop_shape, scale, random_draws, 
                to_augment=True, no_rotation=True, logging=True):
    imgs = []
    if logging:
        print "%s [%d] Processing file %s" % (get_time(), os.getpid(), name)
    pad_value = 127
    img = image_load(name)
    simg = scale_radius(img, round(scale / .9))
    uimg = unsharp_img(simg, round(scale / .9))
    suimg = subsample_inner_circle_img(uimg, round(scale / .9), pad_value)
    cimg = center_crop(suimg, crop_shape)
    pimg = pad_img(cimg, (2 * scale, 2 * scale, 3), value=127)
    pimg[:10, :, :] = pad_value
    pimg[-10:, :, :] = pad_value
    imgs.append(pimg)

    # Check if augmentation is needed
    if (to_augment and np.random.uniform(0, 1) > pb[label]) or (not to_augment):
        return imgs

    for i in range(random_draws):
        dist_img = get_distorted_img(simg, 127, no_rotation)
        uimg = unsharp_img(dist_img, round(scale / .9))
        suimg = subsample_inner_circle_img(uimg, round(scale / .9), pad_value)
        cimg = center_crop(suimg, (256, 256))
        dimg = pad_img(cimg, (2 * scale, 2 * scale, 3), value=127)
        dimg[:10, :, :] = pad_value
        dimg[-10:, :, :] = pad_value
        imgs.append(dimg)

    return imgs


def write_imgs(imgs, name, label, output_folder_name, crop_shape):
    if imgs is None:
        print "%s [%d] WARN: File not processed: %s" % (get_time(), os.getpid(), name)
        return
    new_name = os.path.join(output_folder_name, '%d' % label, 
                            os.path.basename(name))
    make_folder_tree(new_name)
    for i, im in enumerate(imgs):
        if i == 0:
            cv2.imwrite(new_name, im)
            continue
        out_name = new_name.replace(".jpeg", "_%d.jpeg" % (i))
        out_im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_name, out_im)


def main_proc(name, label, out_folder, scale, crop_shape,
              random_draws, to_augment, no_rotation, i):
    ferr = open("out_%d.log" % os.getpid(), 'a')
    sys.stdout = ferr
    sys.stderr = ferr
    imgs = process_img(name, label, crop_shape, scale, random_draws, 
                       to_augment, no_rotation, logging=True)
    write_imgs(imgs, name, label, out_folder, crop_shape)
    if i % 10 == 0:
        print "Processed: %d" % i
    ferr.close()


def main(in_folder, train_folder, val_folder, labels_file,
         scale, crop_shape, random_draws, valsplit):
    names = parse_folder(in_folder, "jpeg")
    print "Total number of files in %s: %d" % (in_folder, len(names))
    pdlab = read_csv(labels_file,
                     names=['image', 'label'], index_col='image', header=0)
    # Determine the train and val split
    pdlab = sample_train_val_split(names, pdlab, valsplit)
    
    # Create a parallel pool
    errf = open('err.log', 'w')
    sys.stderr = errf
    with Parallel(n_jobs=8) as parallel:
        rets = parallel(delayed(main_proc)(
                name,
                pdlab.ix[extract_filename_in_path(name)]['label'],
                train_folder if pdlab.ix[
                    extract_filename_in_path(name)][
                    'val'] == 0 else val_folder,
                scale,
                crop_shape,
                random_draws,
                True if pdlab.ix[
                    extract_filename_in_path(name)][
                    'val'] == 0 else False,
                False,
                i)
                        for i, name in enumerate(names))
        print "Done. A total of %d files processed." % len(rets)
        for f in glob.glob("*.log"):
            os.unlink(f)
    errf.close()


if __name__ == '__main__':
    in_folder = '/home/ubuntu/dataset/train/'
    train_folder = '/home/ubuntu/dataset/train_aug/'
    val_folder = '/home/ubuntu/dataset/val_aug/'
    labels_file = '/home/ubuntu/dataset/labels.txt'
    scale = 128
    crop_shape = (256, 256)
    random_draws = 3
    valsplit = 1500

    main(in_folder, train_folder, val_folder, labels_file,
         scale, crop_shape, random_draws, valsplit)
