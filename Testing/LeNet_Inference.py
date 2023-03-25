#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model


def preprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), 28, 28, 1))
    return array

def noise(array):
    """
    Adds random noise to each image in the supplied array.
    """
    noise_factor = 0.4
    noisy_array = array + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=array.shape
    )
    return np.clip(noisy_array, 0.0, 1.0)

def display(array1, array2):
    """
    Displays ten random images from each one of the supplied arrays.
    """
    n = 10
    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]
    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


# In[3]:


(train_data, train_y), (test_data, test_y) = mnist.load_data()

# Normalize and reshape the data
train_data = preprocess(train_data)
test_data = preprocess(test_data)
# Create a copy of the data with added noise
noisy_train_data = noise(train_data)
noisy_test_data = noise(test_data)

# Display the train data and a version of it with added noise
display(train_data, noisy_train_data)
from tensorflow.keras.utils import to_categorical
y_test = to_categorical(test_y)


# In[41]:


pruned_autoencoder = tf.keras.models.load_model('../models/Lenet_mnist_autoencoder_prunned_model.h5')
prunned_lenet = tf.keras.models.load_model('../models/lenet_mnist_prunned.h5')
lenet_base= tf.keras.models.load_model('../models/Lenet_best.h5')
lenet_combined= tf.keras.models.load_model('Lenet_Mnist_Combined.h5')
lenet_tf_prunned= tf.keras.models.load_model('tensorflow_prunned_Lenet.h5')


# In[14]:


lenet_base.summary()


# In[ ]:


def find_classes(x):
    x_train_subclass=[]
    y_train_subclass=[]
    index = np.where(y_train == x)
    #print(index[0])
    for indices in index[0]:
       x_train_subclass.append(x_train[indices])
       y_train_subclass.append(y_train[indices])
    print(len(x_train_subclass))
    #plt.imshow(x_train_subclass[10].reshape(32,32,3))
    x_train_subclass = np.array(x_train_subclass)
    x_train_subclass= x_train_subclass.reshape(5000,-1)
    print(len(x_train_subclass), len(y_train_subclass))
    return x_train_subclass,y_train_subclass,x


# In[ ]:





# In[ ]:





# In[40]:


#####Inference time using autoencoder
import time

start_inference= time.time()
score = prunned_lenet.evaluate(auto_test, y_test, batch_size=64, verbose=0)
end_inference = time.time()
inference_time= end_inference- start_inference
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print("Total time taken ", inference_time)


# In[70]:


prunned_lenet.summary()


# In[39]:


start= time.time()
score = lenet_base.evaluate(test_data, y_test, batch_size=64, verbose=0)
end = time.time()
print("Inference Time required : ", end-start)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


start= time.time()
score = lenet_base.evaluate(test_data, y_test, batch_size=1, verbose=0)
end = time.time()
print("Inference Time required : ", end-start)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[20]:


pruned_autoencoder.summary()


# In[21]:


pruned_autoencoder.layers[3]._name='encoded'  
layer_output=pruned_autoencoder.get_layer('encoded').output  #get the Output of the Layer
base_encoded=tf.keras.models.Model(inputs=pruned_autoencoder.input,outputs=layer_output)


# In[28]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
# metrics 
from keras.metrics import categorical_crossentropy
# optimization method
from tensorflow.keras.optimizers import SGD


# In[63]:


def LeNet_01():
        model = Sequential() 
        model.add(base_encoded)
        #model.add(Flatten())
        #model.add(Conv2D(filters = 1, kernel_size = (1,1), padding = 'same', activation = 'relu', input_shape = (28,28,1)))
        # Max-pooing layer with pooling window size is 2x2
        #model.add(MaxPooling2D(pool_size = (2,2)))
        #model.add(Flatten())
        # The first fully connected layer 
        # The output layer  
        model.add(Dense(10, activation = 'softmax'))
        # compile the model with a loss function, a metric and an optimizer function
        # In this case, the loss function is categorical crossentropy, 
        # we use Stochastic Gradient Descent (SGD) method with learning rate lr = 0.01 
        # metric: accuracy 
        return model


# In[64]:


lenet_prunned= LeNet_01()
opt = SGD(lr = 0.01)
        
lenet_prunned.compile(loss = categorical_crossentropy, 
                    optimizer = opt, 
                    metrics = ['accuracy']) 
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(train_y)
y_test = to_categorical(test_y)


# In[65]:


lenet_prunned.summary()


# In[20]:


train_data_dimensioned= train_data.reshape(60000,784)
test_data_dimensioned= test_data.reshape(10000,784)


# In[66]:


lenet_prunned.fit(train_data_dimensioned, y_train, batch_size=64, epochs=40, validation_data=(test_data_dimensioned, y_test))


# In[33]:


start= time.time()
score = lenet_combined.evaluate(test_data_dimensioned, y_test, verbose=0)
end = time.time()
print("Inference Time required : ", end-start)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[24]:


lenet_combined.summary()


# In[ ]:


lenet_prunned.sav


# In[12]:


lenet_prunned.summary()


# In[84]:


pip install -U tensorboard_plugin_profile


# In[8]:


options = tf.profiler.experimental.ProfilerOptions(host_tracer_level = 3,
                                                   python_tracer_level = 1,
                                                   device_tracer_level = 1)
tf.profiler.experimental.start('logdir', options = options)
# Training code here
score = lenet_base.evaluate(test_data, y_test, batch_size=64, verbose=0)
tf.profiler.experimental.stop()


# In[4]:


mkdir logs


# In[1]:


logdir= '/home/hmahmud/.jupyter/Old_paper_code/Peng/logs'


# In[9]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[37]:


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

  return os.path.getsize(zipped_file)


# In[38]:


print("Size of gzipped pruned model without stripping: %.2f bytes" % (get_gzipped_model_size(lenet_base)))


# In[13]:


get_ipython().system('kill 3825557')


# In[ ]:


tmp(/tensorboard, --port, 6006)


# In[ ]:




