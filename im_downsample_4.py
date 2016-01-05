#Preprocess training images.
import cv2, glob, numpy as np
from joblib import Parallel, delayed
from utils import files_list, get_time
import sys
import os


root_folder = '/home/ubuntu/dataset/'
run_mode = "train"
# in_folder_train = 'train/'
# in_folder_val = 'validation/'
# in_folder_test = 'test/test/'


def downsample(fname_label, mode, logging=True):
    name, label = fname_label
    if logging:
        print "%s [%d] Processing file %s" % (get_time(), os.getpid(), name)
    a = cv2.imread(name).astype(np.uint8)
    s = 512. / a.shape[1]
    out = cv2.resize(a, None, fx=s, fy=s)
    write_img(out, name, mode)
    return a

def write_img(img, name, mode):
    new_name = name.replace("%s/" % (mode), "%s_512/" % (mode))
    new_dir = os.path.dirname(new_name)
    try:
        os.makedirs(new_dir)
    except OSError:
        pass
    cv2.imwrite(new_name, img)

def main(mode):
    in_fold = os.path.join(root_folder, mode)
    print "Reading images from %s" % in_fold
    print "Run mode: %s" % mode.upper()
    names = files_list(in_fold, mode)
    print "Total number of files: %d" % len(names)
    
    # Create a parallel pool
    errf = open('err.log', 'w')
    sys.stderr = errf
    with Parallel(n_jobs=8) as parallel:
        rets = parallel(delayed(downsample)(fname_label, mode) 
                        for fname_label in names)
        print "Done. A total of %d files processed." % len(rets)
    errf.close()

    
if __name__ == '__main__':
    mode = run_mode
    main(mode)
