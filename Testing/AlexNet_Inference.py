#!/usr/bin/env python
# coding: utf-8

# In[36]:


import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16 
from tensorflow.keras.models import Model
# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import tensorflow as tf
from umap import UMAP
import tensorflow as tf
import hdbscan
import tensorflow as tf
print(tf.__version__)
from keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model


# In[4]:


###I used brancynet data but not normalized , this keras data is same as the data downloaded
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
cifar_x_test= x_test/255.0
cifar_y_test = y_test.flatten()


# In[29]:


import dill
with open("../models/Best_Trained_models/CIFAR10/autoencoder_model_entro_latest1.bn", "rb") as f:
         auto_encoder = dill.load(f)


# In[ ]:





# In[43]:


score = auto_encoder.predict(cifar_x_test,verbose=1)


# In[ ]:





# In[35]:


auto_encoder.summary()


# In[ ]:





# In[44]:


##load the models from the source folder
prunned_AlexNet=tf.keras.models.load_model('../models/Best_Trained_models/CIFAR10/CIFAR10_Encoder_Prunned_AlexNet.h5')
Base_AlexNet= tf.keras.models.load_model('../models/Best_Trained_models/CIFAR10/AlextNet_Original.h5')


# In[58]:


import zipfile
import tempfile
def get_gzipped_model_size(model):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile

  _, keras_file = tempfile.mkstemp('.h5')
  model.save(keras_file, include_optimizer=False)

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(keras_file)
    
  return ((os.path.getsize(zipped_file))/1024)/1024


# In[59]:


print("Size of gzipped pruned model without stripping: %.2f MegaBytes" % (get_gzipped_model_size(Base_AlexNet)))


# In[ ]:





# In[40]:


import time
start = time.time()
score = prunned_AlexNet.evaluate(cifar_x_test, cifar_y_test, batch_size = 1, verbose=0)
end = time.time()
print("Time Taken by Our Model ",end-start)


# In[41]:


import time
start = time.time()
score = Base_AlexNet.predict(cifar_x_test, batch_size = 1, verbose=0)
end = time.time()
print("Time Taken by Base AlexNet ",end-start)


# In[42]:


import tensorflow as tf
tf.version.VERSION


# In[ ]:





# In[37]:


Base_AlexNet.summary()


# In[9]:


prunned_AlexNet.summary()


# In[10]:


prunned_AlexNet.summary()


# In[ ]:




