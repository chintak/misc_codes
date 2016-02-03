
# coding: utf-8

# In[1]:

import os
from utils import parse_folder
from sklearn.svm import OneClassSVM
import numpy as np
import matplotlib.pyplot as plt
import cv2

# get_ipython().magic(u'matplotlib inline')


# In[2]:

from joblib import Parallel, delayed


# In[3]:

in_folder = '/media/shared/dr/DiabeticRetinopathy/train_orig/'
names = np.asarray(parse_folder(in_folder, "jpeg"))


# In[4]:

num_samples = 20000
resize_size = (25, 25)
train_split = 0.9


# In[5]:

idx = np.arange(names.shape[0])
rng = np.random.RandomState(seed=1234)
rng.shuffle(idx)
X_names = names[idx[:num_samples]]


# In[114]:

def load_img(fname, i, tot):
    img = cv2.imread(fname, 0)
    im = cv2.resize(img, resize_size).ravel().astype(np.float32)
#     im[im==0] = im[im!=0].mean()
    im = im / 255.
    if i % 100 == 0:
        print "Loaded %d/%d" % (i, tot)
    return im


# In[115]:

# a = load_img(X_names[0])
# plt.imshow(a.reshape(resize_size))
# plt.colorbar()


# In[116]:

with Parallel(n_jobs=10) as parallel:
    X_unnorm = parallel(delayed(load_img)(fname, i, X_names.shape[0])
                 for i, fname in enumerate(list(X_names)))
    X_unnorm = np.asarray(list(X_unnorm), dtype=np.float32)


# In[117]:

train_samp = int(train_split * num_samples)
X_unnorm_train = X_unnorm[:train_samp]
X_unnorm_val = X_unnorm[train_samp + 1:]


# In[118]:

# X_mean = X_unnorm_train.mean(axis=0)
# X_std = X_unnorm_train.std(axis=0)
# X_train = (X_unnorm_train - X_mean) / X_std
# X_val = (X_unnorm_val - X_mean) / X_std
X_train = X_unnorm_train
X_val = X_unnorm_val


# In[119]:

# a = load_img(X_names[1])
# plt.imshow(X_train[1].reshape(resize_size))


# In[120]:

clf = OneClassSVM(nu=0.01, kernel="rbf", gamma=0.01)


# In[121]:

clf.fit(X_train)


# In[122]:

print "Error training: %d/%d" % (X_train[clf.predict(X_train)==-1].shape[0], X_train.shape[0])
print "Error training: %d/%d" % (X_val[clf.predict(X_val)==-1].shape[0], X_val.shape[0])


# In[108]:

import cPickle as pickle


# In[123]:

pickle.dump(clf, open('full_retinal_img_clf.pkl', 'wb'))


# In[124]:

clf2 = pickle.load(open('full_retinal_img_clf.pkl', 'rb'))


# In[125]:

print "Error training: %d/%d" % (X_train[clf2.predict(X_train)==-1].shape[0], X_train.shape[0])
print "Error training: %d/%d" % (X_val[clf2.predict(X_val)==-1].shape[0], X_val.shape[0])


# In[ ]:



