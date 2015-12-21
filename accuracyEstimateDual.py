from __future__ import division
import numpy as np
from scipy import stats
import caffe
from augmentImages import process_img
from utils import files_list, get_curl_pred, process_one
from joblib import Parallel, delayed
import sys
import tempfile
import os


caffe.set_device(0)
caffe.set_mode_gpu()

scale = 200
crop_shape = (304, 304)
input_shape = (303, 303)
random_draws = 0
batch_size = 5
uniq_im_per_batch = batch_size // (random_draws + 1)


def key_names(f1):
    return int(f1[0].split('/')[-1].split('_')[0])

model_path01 = '/home/ubuntu/model/weight01.caffemodel'
model_path14 = '/home/ubuntu/model/weight14.caffemodel'
dep_path01 = '/home/ubuntu/model/deploy01.prototxt'
dep_path14 = '/home/ubuntu/model/deploy14.prototxt'
mean_img = '/home/ubuntu/model/mean01.jpg'
test_path = '/home/ubuntu/dataset/validation/'

# net01 = caffe.Net(dep_path01, model_path01, caffe.TEST)
# net01.blobs['data'].reshape(*(batch_size, 3, input_shape[0], input_shape[1]))

# net14 = caffe.Net(dep_path14, model_path14, caffe.TEST)
# net14.blobs['data'].reshape(*(batch_size, 3, input_shape[0], input_shape[1]))

# transformer = caffe.io.Transformer({'data': net01.blobs['data'].data.shape})
# transformer.set_mean('data', caffe.io.load_image(mean_img).mean(0).mean(0))
# transformer.set_transpose('data', (2,0,1))
# transformer.set_channel_swap('data', (2,1,0))
# transformer.set_raw_scale('data', 255.0)

names = sorted(files_list(test_path, "val"), key=key_names)
num_files = len(names)
print "Total number of test files: %d" % num_files

accuracy = 0.0
out_file = open("val_pred.csv", "w", 0)
out_file.write("image,lab,pred\n")

for i in xrange(num_files):
    im = (caffe.io.load_image(names[i][0])*255).astype(np.uint8)
    img = process_one(im, crop_shape, scale)
    tmfile = tempfile.mkdtemp() + "file.jpeg"
    caffe.io.skimage.io.imsave(tmfile, img)
    prob, pred = get_curl_pred(tmfile)
    accuracy += names[i][1] == pred
    print names[i][0], pred, prob
    out_file.write("%s,%d\n" % (names[i][0].split('/')[-1].split('.')[0], pred))
    os.unlink(tmfile)

out_file.close()
print "Accuracy on validation dataset: %.2f" % (accuracy/num_files)
print "Done"
