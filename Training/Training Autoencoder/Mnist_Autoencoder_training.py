#!/usr/bin/env python
# coding: utf-8

# In[15]:


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


# In[16]:


(train_data, train_y), (test_data, test_y) = mnist.load_data()

# Normalize and reshape the data
train_data = preprocess(train_data)
test_data = preprocess(test_data)
# Create a copy of the data with added noise
noisy_train_data = noise(train_data)
noisy_test_data = noise(test_data)

# Display the train data and a version of it with added noise
display(train_data, noisy_train_data)


# In[17]:


def calculate_entropy(model,data):
    
    data=np.array(data)
    xdata = data.reshape(-1, 28, 28, 1)
    #ydata = cifar_y_test
    softmax_data = model.predict(xdata)
    entropy_value = np.array([entropy(s) for s in softmax_data])
    ret =0
    #print(entropy_value)
    #print(len(entropy_value))
    minimum = entropy_value[0]
    #print(entropy_value[609])
    for idx in range(1,len(entropy_value)):
        if(minimum>entropy_value[idx]):
            minimum = entropy_value[idx]
            #print(minimum)
            ret=idx

    return  ret


# In[ ]:





# In[ ]:


def calculate_entropy(model,data):
    from scipy.stats import entropy

    data=np.array(data)
    xdata = data.reshape(-1, 32, 32, 3)
    #ydata = cifar_y_test
    softmax_data = model.predict(xdata)
    entropy_value = np.array([entropy(s) for s in softmax_data])
    ret =100
    #print(entropy_value)
    #print(len(entropy_value))
    minimum = entropy_value[0]
    #print(entropy_value[609])
    for idx in range(1,len(entropy_value)):
        if(minimum<entropy_value[idx]):
            minimum = entropy_value[idx]
            #print(minimum)
            ret=idx

    return  ret


# In[ ]:





# In[18]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
# metrics 
from keras.metrics import categorical_crossentropy
# optimization method
from tensorflow.keras.optimizers import SGD


def LeNet():
        model = Sequential()  
        model.add(Conv2D(filters = 5, kernel_size = (5,5), padding = 'same', strides = 1,  activation = 'relu', input_shape = (28,28,1)))
        # Max-pooing layer with pooling window size is 2x2
        model.add(MaxPooling2D(pool_size = (2,2)))
        # Convolutional layer 
        model.add(Conv2D(filters = 10, kernel_size = (5,5), padding = 'same', strides = 1,activation = 'relu'))
        # Max-pooling layer 
        model.add(MaxPooling2D(pool_size = (2,2)))
        # Flatten layer 
        model.add(Conv2D(filters = 20, kernel_size = (5,5), padding = 'same', strides = 1,activation = 'relu'))
        model.add(Flatten())

        # The first fully connected layer 
        model.add(Dense(84, activation = 'relu'))

        # The output layer  
        model.add(Dense(10, activation = 'softmax'))

        # compile the model with a loss function, a metric and an optimizer function
        # In this case, the loss function is categorical crossentropy, 
        # we use Stochastic Gradient Descent (SGD) method with learning rate lr = 0.01 
        # metric: accuracy 

        

        return model


lenet= LeNet()
opt = SGD(lr = 0.01)
        
lenet.compile(loss = categorical_crossentropy, 
                    optimizer = opt, 
                    metrics = ['accuracy']) 
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(train_y)
y_test = to_categorical(test_y)


# In[ ]:





# In[31]:


history1 = lenet.fit(train_data, y_train, batch_size=64, epochs=20, validation_data=(test_data, y_test))


# In[14]:


lenet.summary()


# In[243]:


import time
start= time.time()
score = best_lenet.evaluate(test_data, y_test, batch_size=32, verbose=1)
end = time.time()
print("Inference Time required : ", end-start)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[5]:


def find_classes(x):
    x_train_subclass=[]
    y_train_subclass=[]
    index = np.where(train_y == x)
    #print(index[0])
    for indices in index[0]:
       x_train_subclass.append(train_data[indices])
       y_train_subclass.append(y_train[indices])
    print(len(x_train_subclass))
    #plt.imshow(x_train_subclass[10].reshape(32,32,3))
    x_train_subclass = np.array(x_train_subclass)
    x_train_subclass= x_train_subclass.reshape(-1,784)
    print(len(x_train_subclass), len(y_train_subclass))
    return x_train_subclass,y_train_subclass,x


# In[9]:


total_easy_all=[]
total_hard_all=[]
val_easy_all=[]
val_hard_all=[]
total_reasy_all=[]
val_reasy_all=[]
total_hard=[]
total_easy=[]
x=0
from scipy.stats import entropy
for final_class in range(0,10):
    x_t,t_t, x = find_classes(final_class)
    #labels = Hdb_cluster(x_t,t_t, x)
    #labels = kmeans_clustering(x_t,t_t, x)
    print(x)
    #totaL_ea,total_ha,total_reasy=clustering_bucketing(x_t,labels,x)
    idx=calculate_entropy(lenet,x_t)
    
    print(len(x_t))
    total_hard.append(x_t)
    print(len(total_hard))
    easy=[]
    easy.append(x_t[idx])
    var_easy= easy*(len(x_t))
    total_easy.append(var_easy)
    for itera in range (len(x_t)):
        total_easy_all.append(x_t[idx])
        total_hard_all.append(x_t[itera])
    print(len(total_easy))
    #total_easy_all.append(total_easy)
    #total_hard_all.append(total_hard)    
    for id1 in range (len(x_t)//3):
        val_hard_all.append(x_t[id1])
        val_easy_all.append(x_t[idx])
    #val_hard_all+= total_hard_all[x:len(total_hard_all)//3]
    #val_easy_all+= total_easy_all[x:len(total_easy_all)//3]
    #x+=len(total_hard_all)


# In[10]:


total_hard= total_hard_all+total_hard_all
total_easy= total_easy_all+total_easy_all
val_hard= val_hard_all+val_hard_all
val_easy= val_easy_all+val_easy_all


# In[11]:


total_hard=np.array(total_hard)
total_easy=np.array(total_easy)
val_hard=np.array(val_hard)
val_easy=np.array(val_easy)
total_hard=total_hard.reshape(-1,28,28,1)
total_easy=total_easy.reshape(-1,28,28,1)
val_hard=val_hard.reshape(-1,28,28,1)
val_easy=val_easy.reshape(-1,28,28,1)

print(total_hard.shape)
print(total_easy.shape)
print(val_hard.shape)
print(val_easy.shape)


# In[223]:


from tensorflow.keras import regularizers
#encoding_dim = 512  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
#with batch size 4, encoding_dim 128, validation error  = 0.12
#                           256,                     0.1166
#                           512,                     0.1128
# This is our input image
####for 98.59 accuracy
input_img = tf.keras.Input(shape=(784,))
encoded = layers.Dense(256, activation='relu')(input_img)
encoded = layers.Dense(128, activation='relu')(encoded)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(32, activation='relu')(encoded)

decoded = layers.Dense(32, activation='relu')(encoded)
decoded = layers.Dense(64, activation='relu')(decoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(784, activation='sigmoid')(decoded)


####dont change it 


'''
input_img = tf.keras.Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = layers.Dense(512, activation='relu',activity_regularizer=regularizers.l1(10e-5))(input_img)
encoded = layers.Dense(256, activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(512, activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)
decoded = layers.Dense(784, activation='sigmoid')(decoded)
#decoded = layers.Dense(784, activation='sigmoid')(encoded)
'''

'''
input_img = tf.keras.Input(shape=(784,))
encoded = layers.Dense(784, activation='relu')(input_img)
encoded = layers.Dense(384, activation='relu')(encoded)
#encoded = layers.Dense(32, activation='relu')(encoded)

decoded = layers.Dense(32, activation='linear')(encoded)
#decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(784, activation='softmax')(decoded)
'''


#### for prunned model#########
input_img = tf.keras.Input(shape=(784,))
encoded = layers.Dense(128, activation='relu')(input_img)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(32, activation='relu')(encoded)

decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(784, activation='sigmoid')(decoded)


###########



# This model maps an input to its reconstruction
simple_auto = tf.keras.Model(input_img, decoded)
simple_auto.compile(optimizer='adam', loss='binary_crossentropy')


# In[197]:


hard = total_hard.reshape(-1,784)
easy = total_easy.reshape(-1,784)
V_hard = val_hard.reshape(-1,784)
V_easy = val_easy.reshape(-1,784)


# In[1]:


simple_auto.summary()


# In[86]:


checkpoint_filepath = 'models/Lenet_autoencoder_Mnist_best_mse.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True)


# In[227]:


simple_auto.fit(
    hard,easy,
    epochs=20,
    batch_size=128,
    shuffle=True,
    callbacks= model_checkpoint_callback,
    validation_data=(V_hard, V_easy)
)


# In[208]:


prev =0
epochs=1000
for epoch in range(epochs):

    simple_auto.fit(
    hard,easy,
    epochs=1,
    batch_size=128,
    shuffle=True,
    #callbacks= model_checkpoint_callback,
    validation_data=(V_hard, V_easy))
    
    decoded_images= simple_auto.predict(test)
    auto_test = decoded_images.reshape(-1,28,28,1)
    score = best_lenet.evaluate(auto_test, y_test, batch_size=32, verbose=1)
    
    if(score[1]>prev):
        prev = score[1]
        simple_auto.save('models/autoencoder_mnist_iterative_re.h5')
        


# In[239]:


simple_auto = tf.keras.models.load_model('models/autoencoder_mnist_iterative.h5')
simple_auto = tf.keras.models.load_model('models/autoencoder_mnist_iterative_re.h5')
simple_auto = tf.keras.models.load_model('models/Best_Trained_models/MNIST/autoencoder_mnist_iterative_re.h5')
simple_auto = tf.keras.models.load_model('models/Lenet_mnist_autoencoder_prunned_model.h5')
best_lenet = tf.keras.models.load_model('models/Lenet_best.h5')
prunned_lenet=tf.keras.models.load_model('models/lenet_mnist_prunned.h5')


# In[240]:


import time
test = test_data.reshape(-1,784)
start= time.time()
decoded_images= simple_auto.predict(test)
end = time.time()
print("Autoencoder Time required : ", end-start)
auto_test = decoded_images.reshape(-1,28,28,1)
start= time.time()
score = prunned_lenet.evaluate(auto_test, y_test, batch_size=32, verbose=1)
end = time.time()
print("Inference Time required : ", end-start)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[241]:


simple_auto.summary()


# In[62]:


#simple_auto.save('models/autoencoder_mnist.h5')


# In[219]:



score = prunned_lenet.evaluate(test_data, y_test, batch_size=32, verbose=1)


# In[192]:


simple_auto.summary()


# In[203]:


prunned_lenet.summary()


# In[229]:


simple_auto = tf.keras.models.load_model('models/Lenet_Mnist_autoencoder_simple.h5')


# In[245]:


import time
test = test_data.reshape(-1,784)
start= time.time()
decoded_images= simple_auto.predict(test)
end = time.time()
print("Autoencoder Time required : ", end-start)
auto_test = decoded_images.reshape(-1,28,28,1)
start= time.time()
score = prunned_lenet.evaluate(auto_test, y_test, batch_size=32, verbose=1)
end = time.time()
print("Inference Time required : ", end-start)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[252]:



auto_en = tf.keras.models.load_model('models/Best_Trained_models/MNIST/Lenet_mnist_autoencoder_prunned_model.h5')
test = test_data.reshape(-1,784)
start= time.time()
decoded_images= auto_en.predict(test)
end = time.time()
print("Autoencoder Time required : ", end-start)

check = tf.keras.models.load_model('models/lenet_mnist_prunned.h5')
start= time.time()
score = check.evaluate(auto_test, y_test, batch_size=32, verbose=0)
end = time.time()
print("Inference Time required : ", end-start)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[253]:


best_lenet = tf.keras.models.load_model('models/Lenet_best.h5')
start= time.time()
score = best_lenet.evaluate(test_data, y_test, batch_size=32, verbose=0)
end = time.time()
print("Inference Time required : ", end-start)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[4]:


simple_auto = tf.keras.models.load_model('models/Best_Trained_models/MNIST/autoencoder_mnist_iterative_re.h5')


# In[7]:


simple_auto.summary()


# In[ ]:


get_ipython().system("python -c 'import tensorflow as tf; print(tf.__version__)")


# In[11]:


get_ipython().system('pip list | grep tensorflow')


# In[19]:


pip install nbzip


# In[ ]:




