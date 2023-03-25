#!/usr/bin/env python
# coding: utf-8

# In[10]:


import tempfile
import os

import tensorflow as tf
import numpy as np

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
# metrics 
from keras.metrics import categorical_crossentropy
# optimization method
from tensorflow.keras.optimizers import SGD

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 and 1.
train_images = train_images / 255.0
test_images = test_images / 255.0


model_for_pruning_1=tf.keras.models.load_model('../models/Best_Trained_models/CIFAR10/tf_prunned.h5')
import time
start = time.time()
score = model_for_pruning_1.evaluate(test_images, test_labels, verbose=0)
end = time.time()
print("Time Taken by Pruned Lenet using Tensorflow ",end-start)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[39]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[14]:





# In[16]:





# In[4]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




