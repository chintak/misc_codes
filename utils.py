# -*- coding: utf-8 -*-
from __future__ import division
import cv2
import numpy as np
import os
from os.path import isfile, join
from os import listdir
import math
import sys
import itertools
import urllib
import caffe
from time import gmtime, strftime
from math import cos, sin
from subprocess import check_output, STDOUT
from json import loads


def subsample_inner_circle_img(a, scale, value=0):
    b = np.zeros(a.shape) + value
    cv2.circle(b, (a.shape[1] // 2, a.shape[0] // 2),
               int(scale * 0.9), (1, 1, 1), -1, 8, 0)
    aa = a * b
    aa = aa.astype(np.uint8)
    return aa


def extract_filename_in_path(path):
    return path.split('/')[-1].split('.')[0]


def make_folder_tree(name, is_file=True):
    folder = name if not is_file else os.path.dirname(name)
    if not os.path.isdir(folder):
        os.makedirs(folder)


def image_load(filename):
    return (caffe.io.load_image(filename) * 255.).astype(np.uint8)


def unsharp_img(a, scale):
    b = np.zeros(a.shape)
    cv2.circle(b,(a.shape[1]//2,a.shape[0]//2),int(scale*0.9),(1,1,1),-1,8,0)
    aa = cv2.addWeighted(a,4,cv2.GaussianBlur(a,(0,0),scale/30),-4,128)*b+128*(1-b)
    aa = aa.astype(np.uint8)
    return aa

def process_one(img, crop_shape, scale):
    a = scaleRadius(img, scale)
    ua = unsharp_img(a, scale)
    ca = random_crops(ua, shape=crop_shape)
    return ca

def get_curl_pred(fname, thresh_pb1=0.7):
    res01 = check_output(
        "curl localhost:5000/models/images/classification/classify_one.json -XPOST -F job_id=20151217-070518-faa7 -F image_file=@%s" % (fname), shell=True)
    probs01 = dict(loads(res01)['predictions'])
    res14 = check_output(
        "curl localhost:5000/models/images/classification/classify_one.json -XPOST -F job_id=20151218-081502-a392 -F image_file=@%s" % (fname), shell=True)
    probs14 = dict(loads(res14)['predictions'])
    for k in probs01.iterkeys():
        probs01[k] = round(probs01[k] / 100., 3)
    for k in probs14.iterkeys():
        probs14[k] = round(probs14[k] / 100., 3)
    if probs01['0'] > probs01['1'] or probs01['1'] < thresh_pb1:
        # Best pred level 0
        res = {}
        best_class = 0
        res["level0"] = probs01['0']
        for i in range(1, 4):
            res["level%d" % i] = round(probs01['1'] * probs14['%d' % i], 3)
    else:
        # Best pred level 1, 2, 3 or 4
        res = {}
        best_class = -1
        max_prob = -1.
        res["level0"] = 0.0
        for i in range(1, 4):
            prb = probs14['%d' % i]
            if prb > max_prob:
                max_prob = prb
                best_class = i
            res["level%d" % i] = prb
    return res, best_class

def get_label_prob(probs01, probs14, thresh_pb1=0.7):
    probs = []
    labels = []
    num_imgs = probs01.shape[0]
    for i in range(num_imgs):
        cl01 = probs01[i, :].argmax()
        if cl01 == 1:
            pb1 = probs01[i, 1] > thresh_pb1
            if pb1:
                cl = probs14[i, :].argmax()
                labels.append(cl + 1)
                pdict = {}
                for j in range(probs14.shape[1]):
                    pdict["level_%d" % (j + 1)] = round(probs14[i, j], 4)
                probs.append(pdict)
            else:
                labels.append(0)
                probs.append({"level_0": round(probs01[i, 0], 4)})
        else:
            labels.append(0)
            probs.append({"level_0": round(probs01[i, 0], 4)})
    return probs, labels        

def parse_folder(folder, ext='jpeg'):
    """
    Return a list of tuples (filename, label). label = -1 if mode = 'test' 
    mode = 'val', 'test', 'train'
    """
    names = []
    for r, ds, fs in os.walk(folder):
        for f in fs:
            if ".%s" % ext not in f:
                continue
            names.append(os.path.join(r, f))
    return names

def process_one(img, crop_shape, scale):
    a = scaleRadius(img, scale)
    ua = unsharp_img(a, scale)
    ca = random_crops(ua, shape=crop_shape)
    return ca


def unsharp_img(a, scale):
    b = np.zeros(a.shape)
    cv2.circle(b, (a.shape[1] // 2, a.shape[0] // 2),
               int(scale * 0.9), (1, 1, 1), -1, 8, 0)
    aa = cv2.addWeighted(a, 4, cv2.GaussianBlur(
        a, (0, 0), scale / 30), -4, 128) * b + 128 * (1 - b)
    aa = aa.astype(np.uint8)
    return aa


def files_list(folder, mode):
    """
    Return a list of tuples (filename, label). label = -1 if mode = 'test'
    mode = 'val', 'test', 'train'
    """
    names = []
    for r, ds, fs in os.walk(folder):
        for f in fs:
            if '.jpeg' not in f:
                continue
            if "test" in mode:
                names.append((os.path.join(r, f), -1))
                continue
            label = int(r.strip('/').split('/')[-1])
            names.append((os.path.join(r, f), label))
    return names


def get_time():
    return strftime("%a, %d %b %Y %H:%M:%S", gmtime())


def test_addDir(dir, path):
    if not os.path.exists(path + '/' + dir):
        os.makedirs(path + '/' + dir)
    return path + '/' + dir


def url_imread(url):
    req = urllib.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)  # 'load it as it is'
    return img


def rand_draw():
    r = np.random.uniform(-0.15, 0)
    alpha = np.random.uniform(-3 * np.pi / 180, 3 * np.pi / 180)
    beta = np.random.uniform(-0.1, 0.1) + alpha
    hflip = np.random.randint(2) == 0
    vflip = np.random.randint(2) == 0
    return (r, alpha, beta, hflip, vflip)


def distort(center, param, no_rotation=False):
    r, alpha, beta, hflip, vflip = param
    if no_rotation:
        alpha = 0.
        beta = 0.
    c00 = (1 + r) * cos(alpha)
    c01 = (1 + r) * sin(alpha)
    if hflip:
        c00 *= -1.0
        c01 *= -1.0
    c02 = (1 - c00) * center[0] - c01 * center[1]
    c10 = -(1 - r) * sin(beta)
    c11 = (1 - r) * cos(beta)
    if vflip:
        c10 *= -1.0
        c11 *= -1.0
    c12 = -c10 * center[0] + (1 - c11) * center[1]
    M = np.array([[c00, c01, c02], [c10, c11, c12]], dtype=np.float32)
    return M


def get_distorted_img(im, border_value=128, no_rotation=False):
    if "float" not in im.dtype.name:
        from skimage import img_as_ubyte
        im = img_as_ubyte(im)
    if im.ndim == 3:
        h, w, c = im.shape
        out = np.zeros_like(im)
        param = rand_draw()
        for i in range(c):
            out[:, :, i] = cv2.warpAffine(im[:, :, i], 
                                          distort((w / 2, h / 2), param, no_rotation), 
                                          im[:, :, 0].T.shape, border_value)
    elif im.ndim == 2:
        h, w = im.shape
        out = np.zeros_like(im)
        param = rand_draw()
        for i in range(c):
            out = cv2.warpAffine(im, 
                                 distort((w / 2, h / 2), param, no_rotation), 
                                 im[:, :, 0].shape, border_value)

    return out


def scale_radius(img, scale):
    h, w, _ = img.shape
    assert h > 0 and w > 0, ("Error: scale_radius: Shape of input img:"
                             " (%d, %d)" % (h, w))
    x = img[h // 2, :, :].sum(1)
    r = 0
    for i in range(10, 20, 2):
        r = (x > x.mean() / i).sum() / 2
        if r > 0:
            break
    s = scale * 1.0 / r
    if r <= 0:
        print("%s [%s] %s: Non-positive r = %f detected -"
              " unable to determine scale." % (get_time(),
                                               os.getpid(), "WARN", r))
        s = scale / w
    return cv2.resize(img, (0, 0), fx=s, fy=s)


def pad_img(im, shape, value=0):
    out = np.ones(shape, dtype=im.dtype) * value
    h, w = im.shape[0], im.shape[1]
    ho, wo = shape[0], shape[1]
    rows, cols = np.ogrid[-h // 2:h // 2, -w // 2:w // 2]
    out[rows + (ho + 1) // 2, cols + (wo + 1) // 2] = im
    return out


def random_crops(im, shape=(256, 256)):
    h, w, _ = im.shape
    ho, wo = shape
    if ho > h - 4 or wo > w - 4:
        im = pad_img(im, (max(h, ho + 4), max(w, wo + 4), 3), 128)
    h, w, _ = im.shape
    # Setup multivariate Gaussian sampler
    mean = [h / 2, w / 2]
    sigma = np.eye(2, dtype=np.float)
    sigma[0, 0] = 0.73 * (h - ho) / 2
    sigma[1, 1] = 0.73 * (w - wo) / 2
    y, x = np.round(np.random.multivariate_normal(
        mean, sigma, size=1)[0]).astype(np.uint16)
    y = min(y, h - (ho // 2))
    x = min(x, w - (wo // 2))
    y = max(y, (ho // 2))
    x = max(x, (wo // 2))
    rows, cols = np.ogrid[-ho // 2:ho // 2, -wo // 2:wo // 2]
    # Return appropriate size
    return im[rows + h//2, cols + w//2]


def bbox(im):
    th = 2
    if im is None:
        return None
    a = np.mean(im, axis=2, dtype=np.float).mean(axis=0)
    b = np.mean(im, axis=2, dtype=np.float).mean(axis=1)
    aidx = np.where(a > th)[0]
    bidx = np.where(b > th)[0]
    return im[max(bidx[0] - 1, 0):min(bidx[-1] + 1, im.shape[0]), max(aidx[0] - 1, 0):min(aidx[-1] + 1, im.shape[1])]


def brightness_decrease(im, max_dec=30, min_dec=10):
    br = np.random.randint(min_dec, max_dec)
    im_clone = im.copy().astype(np.int16) - br
    im_clone[im_clone < 0] = 0
    return im_clone.astype(np.uint8)


def brightness_increase(im, max_inc=30, min_inc=10):
    br = np.random.randint(min_inc, max_inc)
    im_clone = im.copy().astype(np.uint16) + br
    im_clone[im_clone > 255] = 255
    im_clone[im == 0] = 0
    return im_clone.astype(np.uint8)


class flushfile(file):

    def __init__(self, f):
        self.f = f

    def write(self, x):
        self.f.write(x)
        self.f.flush()


def create_fixed_image_shape(img, frame_size=(200, 200, 3), random_fill=False, mode='crop'):
    image_frame = None
    if mode == 'fit':
        X1, Y1, _ = frame_size
        if random_fill:
            image_frame = np.asarray(np.random.randint(
                0, high=255, size=frame_size), dtype='uint8')
        else:
            image_frame = np.zeros(frame_size, dtype='uint8')

        X2, Y2 = img.shape[1], img.shape[0]

        if X2 > Y2:
            X_new = X1
            Y_new = int(round(float(Y2 * X_new) / float(X2)))
        else:
            Y_new = Y1
            X_new = int(round(float(X2 * Y_new) / float(Y2)))

        img = cv2.resize(img, (X_new, Y_new))

        X_space_center = ((X1 - X_new) / 2)
        Y_space_center = ((Y1 - Y_new) / 2)

        # print Y_new, X_new, Y_space_center, X_space_center
        image_frame[Y_space_center: Y_space_center + Y_new,
                    X_space_center: X_space_center + X_new, :] = img

    elif mode == 'crop':
        X1, Y1, _ = frame_size
        image_frame = np.zeros(frame_size, dtype='uint8')

        X2, Y2 = img.shape[1], img.shape[0]

        # increase the size of smaller length (width or hegiht)
        if X2 > Y2:
            Y_new = Y1
            X_new = int(round(float(X2 * Y_new) / float(Y2)))
        else:
            X_new = X1
            Y_new = int(round(float(Y2 * X_new) / float(X2)))

        img = cv2.resize(img, (X_new, Y_new))

        X_space_clip = (X_new - X1) / 2
        Y_space_clip = (Y_new - Y1) / 2

        # trim image both top, down, left and right
        if X_space_clip == 0 and Y_space_clip != 0:
            img = img[Y_space_clip:-Y_space_clip, :]
        elif Y_space_clip == 0 and X_space_clip != 0:
            img = img[:, X_space_clip:-X_space_clip]

        if img.shape[0] != X1:
            img = img[1:, :]
        if img.shape[1] != Y1:
            img = img[:, 1:]

        image_frame[:, :] = img
    return image_frame


def reshape_image(img, frame_size=(200, 200, 3), mode='crop'):
    image = None
    if mode == 'fit':
        X1, Y1, _ = frame_size
        X2, Y2 = img.shape[1], img.shape[0]

        if X2 > Y2:
            X_new = X1
            Y_new = int(round(float(Y2 * X_new) / float(X2)))
        else:
            Y_new = Y1
            X_new = int(round(float(X2 * Y_new) / float(Y2)))

        img = cv2.resize(img, (X_new, Y_new))

        X_space_center = ((X1 - X_new) / 2)
        Y_space_center = ((Y1 - Y_new) / 2)

        image = img

    elif mode == 'crop':
        X1, Y1, _ = frame_size

        X2, Y2 = img.shape[1], img.shape[0]

        # increase the size of smaller length (width or hegiht)
        if X2 > Y2:
            Y_new = Y1
            X_new = int(round(float(X2 * Y_new) / float(Y2)))
        else:
            X_new = X1
            Y_new = int(round(float(Y2 * X_new) / float(X2)))

        img = cv2.resize(img, (X_new, Y_new))

        X_space_clip = (X_new - X1) / 2
        Y_space_clip = (Y_new - Y1) / 2

        # trim image both top, down, left and right
        if X_space_clip == 0 and Y_space_clip != 0:
            img = img[Y_space_clip:-Y_space_clip, :]
        elif Y_space_clip == 0 and X_space_clip != 0:
            img = img[:, X_space_clip:-X_space_clip]

        if img.shape[0] != X1:
            img = img[1:, :]
        if img.shape[1] != Y1:
            img = img[:, 1:]

        image = img
    return image


def generate_window_locations(center, patch_shape, stride=0.5, grid_shape=5):

    assert(grid_shape % 2 != 0), "grid_shape should be odd number"

    assert(stride != 0), "stride should not be <= 0"

    center_y, center_x = center
    string = ""
    mapping = {}

    # left is represented as -, center as 0  and right is +
    if grid_shape % 2 != 0:
        pointer = xrange(-(grid_shape / 2), (grid_shape / 2) + 1)
    # else:
    # 	pointer = range(-(grid_shape/2)+1, (grid_shape/2)+1)

    for i, n, in enumerate(pointer):
        mapping[i] = n
        string += str(i)
    windows_list = []
    sequences = np.asarray(
        list(itertools.product(string, repeat=2)), dtype="int32")

    # if grid_shape%2 != 0:
    # 	center_y = (center_y - (stride * patch_shape[0])/2.0)
    # 	center_x = (center_x - (stride * patch_shape[1])/2.0)

    for y, x in sequences:
        new_center_y, new_center_x = center_y + \
            patch_shape[0] * mapping[y] * stride, center_x + \
            patch_shape[1] * mapping[x] * stride

        res = np.asarray([(math.floor(new_center_y) - patch_shape[0] / 2, math.floor(new_center_x) - patch_shape[1] / 2),
                          (math.ceil(new_center_y) + patch_shape[0] / 2, math.ceil(new_center_x) + patch_shape[1] / 2)], dtype="int32")

        windows_list.append(res)

    return np.asarray(windows_list).tolist()


def getImmediateSubdirectories(dir):
    """
            this function return the immediate subdirectory list
            eg:
                    dir
                            /subdirectory1
                            /subdirectory2
                            .
                            .
            return ['subdirectory1',subdirectory2',...]
    """

    return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]


def getFiles(dir_path):
    """getFiles : gets the file in specified directory
    dir_path: String type
    dir_path: directory path where we get all files
    """
    onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    return onlyfiles


def get_num_batch(data_size, batch_size):
    if data_size % batch_size == 0:
        return data_size / batch_size
    return (data_size / batch_size) + 1


def feature_normalization(data, type='standardization', params=None):
    u"""
            data:
                    an numpy array
            type:
                    (standardization, min-max)
            params {default None}:
                    dictionary
                    if params is provided it is used as mu and sigma when type=standardization else Xmax, Xmin when type=min-max
                    rather then calculating those paramsanter
            two type of normalization
            1) standardization or (Z-score normalization)
                    is that the features will be rescaled so that they'll have the properties of a standard normal distribution with
                            μ = 0 and σ = 1
                    where μ is the mean (average) and σ is the standard deviation from the mean
                            Z = (X - μ)/σ
                    return:
                            Z, μ, σ
            2) min-max normalization
                    the data is scaled to a fixed range - usually 0 to 1.
                    The cost of having this bounded range - in contrast to standardization - is that we will end up with smaller standard
                    deviations, which can suppress the effect of outliers.
                    A Min-Max scaling is typically done via the following equation:
                            Z = (X - Xmin)/(Xmax-Xmin)
                    return Z, Xmax, Xmin
    """
    if type == 'standardization':
        if params is None:
            mu = np.mean(data, axis=0)
            sigma = np.std(data, axis=0)
        else:
            mu = params['mu']
            sigma = params['sigma']
        Z = (data - mu) / sigma
        return Z, mu, sigma

    elif type == 'min-max':
        if params is None:
            Xmin = np.min(data, axis=0)
            Xmax = np.max(data, axis=0)
        else:
            Xmin = params['Xmin']
            Xmax = params['Xmax']

        Xmax = Xmax.astype('float')
        Xmin = Xmin.astype('float')
        Z = (data - Xmin) / (Xmax - Xmin)
        return Z, Xmax, Xmin
