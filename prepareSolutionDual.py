from __future__ import division
import numpy as np
from scipy import stats
import caffe
from augmentImages import process_img
from utils import files_list, get_curl_pred, process_one
from joblib import Parallel, delayed
import os
import tempfile


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
test_path = '/home/ubuntu/dataset/test/test/'
job_id = "model_303_01_14.csv"

# net01 = caffe.Net(dep_path01, model_path01, caffe.TEST)
# net01.blobs['data'].reshape(*(batch_size, 3, input_shape[0], input_shape[1]))

# net14 = caffe.Net(dep_path14, model_path14, caffe.TEST)
# net14.blobs['data'].reshape(*(batch_size, 3, input_shape[0], input_shape[1]))

# transformer = caffe.io.Transformer({'data': net01.blobs['data'].data.shape})
# transformer.set_mean('data', (caffe.io.load_image(mean_img)*255).mean(0).mean(0))
# transformer.set_transpose('data', (2,0,1))
# transformer.set_channel_swap('data', (2,1,0))
# transformer.set_raw_scale('data', 255.0)

names = sorted(files_list(test_path, "test"), key=key_names)
num_files = len(names)
print "Total number of test files: %d" % num_files

out_file = open("pred_%s.csv" % (job_id), "w", 0)

# with Parallel(n_jobs=5) as parallel:
#     for i in xrange(0, num_files, uniq_im_per_batch):
#         upper_idx = min(i + uniq_im_per_batch, num_files)
#         files_batch = names[i:upper_idx]
#         num_uniq_im = (upper_idx - i)
#         ret = parallel(delayed(process_img)(fname_lab, crop_shape, scale, random_draws, "test", False) 
#                        for fname_lab in files_batch)
#         ret = np.asarray(ret).reshape((num_uniq_im * (random_draws + 1), crop_shape[0], crop_shape[1], 3))
#         if ret.shape[0] < batch_size:
#             pad = np.zeros((batch_size - num_uniq_im, crop_shape[0], crop_shape[1], 3), dtype=ret.dtype)
#             ret = np.vstack((ret, pad))
#         assert ret.shape[0] == batch_size, ("Error: the input batch has lesser number"
#                                             " of images than the expected batch size %d" % (batch_size))
#         l = []
#         for j in range(batch_size):
#             l.append(transformer.preprocess('data', ret[j, :input_shape[0], :input_shape[1], :]))
#         net01.blobs['data'].data[:] = np.asarray(l)
#         out01 = net01.forward()
        
#         net14.blobs['data'].data[:] = np.asarray(l)
#         out14 = net14.forward()
        
#         probs01 = out01['prob'][:num_uniq_im * (random_draws + 1), :]
#         probs14 = out14['prob'][:num_uniq_im * (random_draws + 1), :]
#         print probs01[1,:]
#         print probs14[1,:]
#         _, preds = get_label_prob(probs01, probs14, thresh_pb1=0.7)
#         print preds[1]
        
#         preds = np.asarray(preds)
#         preds_im = stats.mode(preds.reshape((num_uniq_im, (random_draws + 1))), axis=1)[0]
#         for j in range(num_uniq_im):
#             out_file.write("%s,%d\n" % (files_batch[j][0].split('/')[-1].split('.')[0], preds_im[j]))
#             print files_batch[j], preds_im[j]
#             caffe.io.skimage.io.imsave(files_batch[j][0].replace("test/test/", ""), l[j].transpose([1, 2, 0]))

for i in xrange(num_files):
    im = (caffe.io.load_image(names[i][0])*255).astype(np.uint8)
    img = process_one(im, crop_shape, scale)
    tmfile = tempfile.mkdtemp() + "file.jpeg"
    caffe.io.skimage.io.imsave(tmfile, img)
    prob, pred = get_curl_pred(tmfile)
    out_file.write("%s,%d\n" % (names[i][0].split('/')[-1].split('.')[0], pred))
    os.unlink(tmfile)
    
out_file.close()
print "Done"
