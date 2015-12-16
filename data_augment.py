import cv2
import numpy as np
from utils import *
from scipy.ndimage.interpolation import rotate


def data_augmentation(orignial_img, frame_size=(256, 256, 3), size_factor_up=1.25, size_factor_down=0.85):
	result = []
	img = create_fixed_image_shape(orignial_img, frame_size, random_fill=False, mode='fit')
	res_img = img.copy()
	result.append(img)
	#img = create_fixed_image_shape(orignial_img, frame_size, random_fill=False, mode='crop')
	#result.append(img)

	#rotate image 30 45 -30 -45, or random rotation
	img_30 = bbox(rotate(orignial_img, np.random.randint(3, high=10)))
	img = create_fixed_image_shape(img_30, frame_size, random_fill=False, mode='fit')
	result.append(img)
	# img = create_fixed_image_shape(img_30, frame_size, random_fill=False, mode='crop')
	# result.append(img)

	img_45 = bbox(rotate(orignial_img, np.random.randint(7, high=15)))
	img = create_fixed_image_shape(img_45, frame_size, random_fill=False, mode='fit')
	result.append(img)
	# img = create_fixed_image_shape(img_45, frame_size, random_fill=False, mode='crop')
	# result.append(img)

	img_m30 = bbox(rotate(orignial_img, np.random.randint(-10, high=-3)))
	img = create_fixed_image_shape(img_m30, frame_size, random_fill=False, mode='fit')
	result.append(img)
	# img = create_fixed_image_shape(img_m30, frame_size, random_fill=False, mode='crop')
	# result.append(img)

	img_m45 = bbox(rotate(orignial_img, np.random.randint(-15, high=-7)))
	img = create_fixed_image_shape(img_m45, frame_size, random_fill=False, mode='fit')
	result.append(img)
	# img = create_fixed_image_shape(img_m45, frame_size, random_fill=False, mode='crop')
	# result.append(img)

	#scale up by factor of 50%(size_factor_up) more
	img_large = cv2.resize(orignial_img, (int(orignial_img.shape[1]*size_factor_up),int(orignial_img.shape[0]*size_factor_up)))
	center = (img_large.shape[0]/2, img_large.shape[1]/2)
	y, x = (center[0]-orignial_img.shape[0]/2, center[1]-orignial_img.shape[1]/2)
	y1, x1 = (center[0]+orignial_img.shape[0]/2, center[1]+orignial_img.shape[1]/2)
	img_large = img_large[y:y1, x:x1]
	img = create_fixed_image_shape(img_large, frame_size, random_fill=False, mode='fit')
	result.append(img)

	#img = create_fixed_image_shape(img_large, frame_size, random_fill=False, mode='crop')
	#result.append(img)

	#scale down by factor of 50%(size_factor_down) more
	img_small = cv2.resize(orignial_img, (int(orignial_img.shape[1]*size_factor_down),int(orignial_img.shape[0]*size_factor_down)))
	center = (orignial_img.shape[0]/2, orignial_img.shape[1]/2)
	y, x = (center[0]-img_small.shape[0]/2, center[1]-img_small.shape[1]/2)
	y1, x1 = (center[0]+img_small.shape[0]/2, center[1]+img_small.shape[1]/2)
	img_frame = np.zeros(orignial_img.shape, dtype='uint8')
	img_frame[y:y1, x:x1, :] = img_small[:y1-y,:x1-x] 
	img = create_fixed_image_shape(img_frame, frame_size, random_fill=False, mode='fit')
	result.append(img)
    
	# Create brighter and darker variations of the image
	img = brightness_increase(res_img, 40, 10)
	result.append(img)
    
	img = brightness_decrease(res_img, 40, 10)
	result.append(img)

	#img = create_fixed_image_shape(img_frame, frame_size, random_fill=False, mode='crop')
	#result.append(img)

	return result
