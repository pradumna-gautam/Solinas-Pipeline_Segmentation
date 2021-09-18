# -*- coding: utf-8 -*-
"""
Created on Tue Aug  31 23:30:09 2021

@author: PRADUMNA
"""
import tensorflow as tf; print(tf.__version__)



import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import shutil
from collections import Counter
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
import matplotlib.image as mpimg
import random, os
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D,\
    Dense, Layer, Reshape, InputLayer, Flatten, Input, MaxPooling2D
from alibi_detect.od import OutlierAE
from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_image



'''def img_to_np(path, resize = True):  
    img_array = []
    fpaths = glob.glob(path, recursive=True)
    for fname in fpaths:
        img = Image.open(fname).convert("RGB")
        if(resize): img = img.resize((64,64))
        img_array.append(np.asarray(img))
    images = np.array(img_array)
    return images'''





#setting the path to the directory containing the pics
path_train = r"C:\Users\kgaut\Documents\Lightshot\Solinas\Datasets\train"
path_test = r"C:\Users\kgaut\Documents\Lightshot\Solinas\Datasets\test"
#appending the pics to the training data list
def training_data(path, resize = True): 
    training_data = []
    for img in os.listdir(path_train):
     pic = cv2.imread(os.path.join(path_train,img))
     pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
     pic = cv2.resize(pic,(64,64))
     training_data.append(np.asarray(pic))
     images = np.array(training_data)
     return images
 
    
'''
#converting the list to numpy array and saving it to a file using #numpy.save
np.save(os.path.join(path_train,'features'),np.array(training_data))

#loading the saved file once again
import numpy as np
# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


saved = np.load(os.path.join(path_train,'features.npy'))

plt.imshow(saved[0].reshape(80,80,3))
plt.imshow(np.array(training_data[0]).reshape(80,80,3))'''


#appending the pics to the test data list
test_data = []
def test_data(path, resize = True): 
    test_data = []
    for img in os.listdir(path_test):
     pic = cv2.imread(os.path.join(path_test,img))
     pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
     pic = cv2.resize(pic,(64,64))
     test_data.append(np.asarray(pic))
     images = np.array(test_data)
     return images


'''
#converting the list to numpy array and saving it to a file using #numpy.save
np.save(os.path.join(path_test,'features'),np.array(test_data))

#loading the saved file once again
saved = np.load(os.path.join(path_test,'features.npy'))

plt.imshow(saved[0].reshape(80,80,3))
plt.imshow(np.array(test_data[0]).reshape(80,80,3))'''



train = training_data(path_train)
test = test_data(path_test)
train = train.astype('float32') / 255.
test = test.astype('float32') / 255.



print(train)
print(test)
print(train.size)
print(test.shape)
print(train.shape)
print(test.shape)


encoding_dim = 1024
dense_dim = [8, 8, 128]

encoder_net = tf.keras.Sequential(
  [
      InputLayer(input_shape=train[0].shape),
      Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu),
      Flatten(),
      Dense(encoding_dim,)
  ])

decoder_net = tf.keras.Sequential(
  [
      InputLayer(input_shape=(encoding_dim,)),
      Dense(np.prod(dense_dim)),
      Reshape(target_shape=dense_dim),
      Conv2DTranspose(256, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2DTranspose(64, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')
  ])

od = OutlierAE( threshold = 0.001,
                encoder_net=encoder_net,
                decoder_net=decoder_net)

adam = tf.keras.optimizers.Adam(learning_rate=1e-4)

od.fit(train, epochs=100, verbose=True,
       optimizer = adam)

od.infer_threshold(test, threshold_perc=95)

preds = od.predict(test, outlier_type='instance',
            return_instance_score=True,
            return_feature_score=True)





for i, fpath in enumerate(glob.glob(path_test)):
    if(preds['data']['is_outlier'][i] == 1):
        source = fpath
        shutil.copy(source, 'img\\')
        
filenames = [os.path.basename(x) for x in glob.glob(path_test, recursive=True)]

dict1 = {'Filename': filenames,
     'instance_score': preds['data']['instance_score'],
     'is_outlier': preds['data']['is_outlier']}
     
df = pd.DataFrame(dict1)
df_outliers = df[df['is_outlier'] == 1]

print(df_outliers)
