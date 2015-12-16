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
from time import gmtime, strftime
from math import cos, sin


def get_time():
    return strftime("%a, %d %b %Y %H:%M:%S", gmtime())

def test_addDir(dir, path):
	if not os.path.exists(path+'/'+dir):
		os.makedirs(path+'/'+dir)
	return path+'/'+dir

def url_imread(url):
	req = urllib.urlopen(url)
	arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
	img = cv2.imdecode(arr,-1) # 'load it as it is'
	return img

def rand_draw():
    r = np.random.uniform(-0.1, 0.1)
    alpha = np.random.uniform(-np.pi / 4, np.pi / 4)
    beta = np.random.uniform(-0.2, 0.2) + alpha
    hflip = np.random.randint(2) == 0
    vflip = np.random.randint(2) == 0
    return (r, alpha, beta, hflip, vflip)

def distort(center, param):
    r, alpha, beta, hflip, vflip = param
    c00 = (1+r) * cos(alpha)
    c01 = (1+r) * sin(alpha)
    if hflip:
        c00 *= -1.0
        c01 *= -1.0
    c02 = (1 - c00) * center[0] - c01 * center[1]
    c10 = -(1-r) * sin(beta)
    c11 = (1-r) * cos(beta)
    if vflip:
        c10 *= -1.0
        c11 *= -1.0
    c12 = -c10 * center[0] + (1 - c11) * center[1]
    M = np.array([[c00, c01, c02], [c10, c11, c12]], dtype=np.float32)
    return M

def get_distorted_img(im):
    if "float" not in im.dtype.name:
        from skimage import img_as_ubyte
        im = img_as_ubyte(im)
    if im.ndim == 3:
        h, w, c = im.shape
        out = np.zeros_like(im)
        param = rand_draw()
        for i in range(c):
            out[:, :, i] = cv2.warpAffine(im[:, :, i], distort((w/2, h/2), param), im[:,:,0].T.shape, borderValue=128)
    elif im.ndim == 2:
        h, w = im.shape
        out = np.zeros_like(im)
        param = rand_draw()
        for i in range(c):
            out = cv2.warpAffine(im, distort((w/2, h/2), param), im[:,:,0].shape, borderValue=128)

    return out

def scaleRadius(img, scale):
    print "%s [%s] %s: Input image size (%d, %d, %d)" % (get_time(), os.getpid(), "LOG", img.shape[0], img.shape[1], img.shape[2])
    assert img.shape[0] > 0 and img.shape[1] > 0, "Error: scaleRadius: Shape of input img: (%d, %d)" % (img.shape[0], img.shape[1])
    x=img[img.shape[0]/2,:,:].sum(1)
    r=(x>x.mean()/10).sum()/2
    if r <= 0:
        return None
    s=scale*1.0/r
    print "%s [%s] %s: r = %f and s = %f, scale = %d" % (get_time(), os.getpid(), "LOG", r, s, scale)
#     assert s <= 1., "%s [%s] Error: scaleRadius: s = %f" % (get_time(), os.getpid(), s)
    try:
        return cv2.resize(img,(0,0),fx=s,fy=s)
    except e:
        print e
        exit(0)

def pad_img(im, shape, value=0):
    out = np.ones(shape, dtype=im.dtype) * value
    h, w = im.shape[0], im.shape[1]
    ho, wo = shape[0], shape[1]
    rows, cols = np.ogrid[-h//2:h//2, -w//2:w//2]
    out[rows + (ho+1)//2, cols + (wo+1)//2] = im
    return out

def random_crops(im, shape=(256, 256)):
    h, w, _ = im.shape
    ho, wo = shape
    if ho > h - 4 or wo > w - 4:
        im = pad_img(im, (max(h, ho + 4), max(w, wo + 4), 3), 128)
    h, w, _ = im.shape
    # Setup multivariate Gaussian sampler
    mean = [h/2, w/2]
    sigma = np.eye(2, dtype=np.float)
    sigma[0, 0] = 0.73 * (h - ho) / 2
    sigma[1, 1] = 0.73 * (w - wo) / 2
    y, x = np.round(np.random.multivariate_normal(mean, sigma, size=1)[0]).astype(np.uint16)
    y = min(y, h - (ho//2))
    x = min(x, w - (wo//2))
    y = max(y, (ho//2))
    x = max(x, (wo//2))
    rows, cols = np.ogrid[-ho//2:ho//2,-wo//2:wo//2]
    # Return appropriate size
    return im[rows + y, cols + x]

def bbox(im):
    th = 2
    if im is None:
        return None
    print "%s [%d] LOG: bbox: Shape (%d, %d, %d)" % (get_time(), os.getpid(), im.shape[0], im.shape[1], im.shape[2])
    a = np.mean(im, axis=2, dtype=np.float).mean(axis=0)
    b = np.mean(im, axis=2, dtype=np.float).mean(axis=1)
    aidx = np.where(a > th)[0]
    bidx = np.where(b > th)[0]
    return im[max(bidx[0]-1, 0):min(bidx[-1]+1, im.shape[0]), max(aidx[0]-1, 0):min(aidx[-1]+1, im.shape[1])]

def brightness_decrease(im, max_dec=30, min_dec=10):
    br = np.random.randint(min_dec, max_dec)
    im_clone = im.copy().astype(np.int16) - br
    im_clone[im_clone<0] = 0
    return im_clone.astype(np.uint8)

def brightness_increase(im, max_inc=30, min_inc=10):
    br = np.random.randint(min_inc, max_inc)
    im_clone = im.copy().astype(np.uint16) + br
    im_clone[im_clone>255] = 255
    im_clone[im==0] = 0
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
			image_frame = np.asarray(np.random.randint(0, high=255, size=frame_size), dtype='uint8')
		else:
			image_frame = np.zeros(frame_size, dtype='uint8')

		X2, Y2 = img.shape[1], img.shape[0]

		if X2 > Y2:
			X_new = X1
			Y_new = int(round(float(Y2*X_new)/float(X2)))
		else:
			Y_new = Y1
			X_new = int(round(float(X2*Y_new)/float(Y2)))

		img = cv2.resize(img, (X_new, Y_new))

		X_space_center = ((X1 - X_new)/2)
		Y_space_center = ((Y1 - Y_new)/2)

		# print Y_new, X_new, Y_space_center, X_space_center
		image_frame[Y_space_center: Y_space_center+Y_new, X_space_center: X_space_center+X_new, :] = img
		
	elif mode == 'crop':
		X1, Y1, _ = frame_size
		image_frame = np.zeros(frame_size, dtype='uint8')

		X2, Y2 = img.shape[1], img.shape[0]

		#increase the size of smaller length (width or hegiht)
		if X2 > Y2:
			Y_new = Y1
			X_new = int(round(float(X2*Y_new)/float(Y2)))
		else:
			X_new = X1
			Y_new = int(round(float(Y2*X_new)/float(X2)))

		img = cv2.resize(img, (X_new, Y_new))

		
		X_space_clip = (X_new - X1)/2
		Y_space_clip = (Y_new - Y1)/2

		#trim image both top, down, left and right
		if X_space_clip == 0 and Y_space_clip != 0:
			img = img[Y_space_clip:-Y_space_clip, :]
		elif Y_space_clip == 0 and X_space_clip != 0:
			img = img[:, X_space_clip:-X_space_clip]

		if img.shape[0] != X1:
			img = img[1:, :]
		if img.shape[1] != Y1:
			img = img[:, 1:]

		image_frame[: , :] = img
	return image_frame

def reshape_image(img, frame_size=(200, 200, 3), mode='crop'):
	image = None
	if mode == 'fit':
		X1, Y1, _ = frame_size
		X2, Y2 = img.shape[1], img.shape[0]

		if X2 > Y2:
			X_new = X1
			Y_new = int(round(float(Y2*X_new)/float(X2)))
		else:
			Y_new = Y1
			X_new = int(round(float(X2*Y_new)/float(Y2)))

		img = cv2.resize(img, (X_new, Y_new))

		X_space_center = ((X1 - X_new)/2)
		Y_space_center = ((Y1 - Y_new)/2)

		image = img

	elif mode == 'crop':
		X1, Y1, _ = frame_size

		X2, Y2 = img.shape[1], img.shape[0]

		#increase the size of smaller length (width or hegiht)
		if X2 > Y2:
			Y_new = Y1
			X_new = int(round(float(X2*Y_new)/float(Y2)))
		else:
			X_new = X1
			Y_new = int(round(float(Y2*X_new)/float(X2)))

		img = cv2.resize(img, (X_new, Y_new))

		
		X_space_clip = (X_new - X1)/2
		Y_space_clip = (Y_new - Y1)/2

		#trim image both top, down, left and right
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

	assert(grid_shape%2 != 0), "grid_shape should be odd number"

	assert(stride != 0), "stride should not be <= 0"

	center_y, center_x = center
	string = ""
	mapping = {}

	# left is represented as -, center as 0  and right is +
	if grid_shape%2 != 0:
		pointer = xrange(-(grid_shape/2), (grid_shape/2)+1)
	# else:
	# 	pointer = range(-(grid_shape/2)+1, (grid_shape/2)+1)

	for i, n, in enumerate(pointer):
		mapping[i] = n
		string += str(i)
	windows_list = []	
	sequences = np.asarray(list(itertools.product(string, repeat=2)), dtype="int32")

	# if grid_shape%2 != 0:
	# 	center_y = (center_y - (stride * patch_shape[0])/2.0)
	# 	center_x = (center_x - (stride * patch_shape[1])/2.0)

	for y, x in sequences:
		new_center_y, new_center_x = center_y +patch_shape[0]*mapping[y]*stride, center_x+patch_shape[1]*mapping[x]*stride

		res =  np.asarray([(math.floor(new_center_y)-patch_shape[0]/2,math.floor(new_center_x)-patch_shape[1]/2), 
		(math.ceil(new_center_y)+patch_shape[0]/2, math.ceil(new_center_x)+patch_shape[1]/2)], dtype="int32")

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
    onlyfiles = [ f for f in listdir(dir_path) if isfile(join(dir_path, f)) ]
    return onlyfiles

def get_num_batch(data_size, batch_size):
	if data_size%batch_size == 0:
		return data_size/batch_size
	return (data_size/batch_size) + 1

def feature_normalization(data, type='standardization', params = None):
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
			sigma =  np.std(data, axis=0)
		else:
			mu = params['mu']
			sigma = params['sigma']
		Z = (data - mu)/sigma
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
		Z = (data - Xmin)/(Xmax - Xmin)
		return Z, Xmax, Xmin

