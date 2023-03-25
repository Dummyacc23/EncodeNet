#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:





# In[ ]:





# In[ ]:





# In[9]:





# In[10]:





# In[ ]:





# In[12]:


get_ipython().system('pip2 install virtualenv')


# In[13]:





# In[14]:





# In[16]:





# In[18]:





# In[19]:





# In[ ]:





# In[2]:


import numpy as np
from keras.models import Sequential
from keras.layers import MaxPooling2D
 
# define input image
image = np.array([[2, 2, 7, 3],
                  [9, 4, 6, 1],
                  [8, 5, 2, 4],
                  [3, 1, 2, 6]])
image = image.reshape(1, 4, 4, 1)
 
# define model containing just a single max pooling layer
model = Sequential(
    [MaxPooling2D(pool_size = 3, strides = 1)])
 
# generate pooled output
output = model.predict(image)
 
# print output image
output = np.squeeze(output)
print(output)


# In[ ]:





# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
X1 = [3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8]
X2 = [5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3]
plt.scatter(X1,X2)
plt.show()


# In[6]:


import pandas as pd
import numpy as np
images = np.array(X1)
label = np.array(X2)
dataset = pd.DataFrame({'label': label,'images': list(images)})


# In[ ]:





# In[ ]:





# In[9]:


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
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import pickle

"""
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 
training images and 10000 test images.
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains 
exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random 
order, but some training batches may contain more images from one class than another. Between them, the training 
batches contain exactly 5000 images from each class.
"""


def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar_10_data(data_dir, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    # get the meta_data_dict
    # num_cases_per_batch: 1000
    # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # num_vis: :3072

    meta_data_dict = unpickle(data_dir + "/cifar-10-batches-py/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    # cifar_train_data_dict
    # 'batch_label': 'training batch 5 of 5'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    # test data
    # cifar_test_data_dict
    # 'batch_label': 'testing batch 1 of 1'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return cifar_train_data, cifar_train_filenames, cifar_train_labels,         cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names


if __name__ == "__main__":
    """show it works"""
    data_dir ="branchynet/B_net/datasets/data/cifar10"
    cifar_10_dir = 'cifar-10-batches-py'

    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names =         load_cifar_10_data(data_dir)

    print("Train data: ", train_data.shape)
    print("Train filenames: ", train_filenames.shape)
    print("Train labels: ", train_labels.shape)
    print("Test data: ", test_data.shape)
    print("Test filenames: ", test_filenames.shape)
    print("Test labels: ", test_labels.shape)
    print("Label names: ", label_names.shape)

    # Don't forget that the label_names and filesnames are in binary and need conversion if used.

    # display some random training images in a 25x25 grid
    num_plot = 5
    f, ax = plt.subplots(num_plot, num_plot)
    for m in range(num_plot):
        for n in range(num_plot):
            idx = np.random.randint(0, train_data.shape[0])
            ax[m, n].imshow(train_data[idx])
            ax[m, n].get_xaxis().set_visible(False)
            ax[m, n].get_yaxis().set_visible(False)
    f.subplots_adjust(hspace=0.1)
    f.subplots_adjust(wspace=0)
    plt.show()


# In[12]:


x_train, train_filenames, y_train, x_test, test_filenames, y_test, label_names =         load_cifar_10_data(data_dir)


# In[13]:


print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


# In[15]:


cifar_x_train=x_train
cifar_x_test =x_test
cifar_y_train = y_train
cifar_y_test = y_test
cifar_x_train, cifar_x_test = x_train / 255.0, x_test / 255.0
 
# flatten the label values
cifar_y_train, cifar_y_test = y_train.flatten(), y_test.flatten()
from sklearn.cluster import SpectralClustering


# In[14]:


fig, ax = plt.subplots(5, 5)
k = 0
 
for i in range(5):
    for j in range(5):
        ax[i][j].imshow(x_train[k], aspect='auto')
        k += 1
 
plt.show()


# In[110]:


def find_classes(x):
    x_train_subclass=[]
    y_train_subclass=[]
    index = np.where(y_train == x)
    #print(index[0])
    for indices in index[0]:
       x_train_subclass.append(cifar_x_train[indices])
       y_train_subclass.append(cifar_y_train[indices])
    #print(len(x_train_subclass))
    #plt.imshow(x_train_subclass[10].reshape(32,32,3))
    x_train_subclass = np.array(x_train_subclass)
    x_train_subclass= x_train_subclass.reshape(5000,3072)
    print(len(x_train_subclass), len(y_train_subclass))
    
    return x_train_subclass,y_train_subclass,x


# In[141]:


from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16


# In[142]:


def extract_features(file, model):
    # load the image as a 224x224 array
    #img = load_img(file, target_size=(32,32))
    # convert from 'PIL.Image.Image' to numpy array
    
    img = np.array(file) 
    #print(img.shape)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,32,32,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    #print(imgx.shape)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    #print(features.shape)
    return features


# In[143]:


def kmeans_clustering(x_train_subclass,y_train_subclass,x):
      model = VGG16(weights="imagenet", include_top=False,input_tensor= tf.keras.layers.Input(shape=(32, 32, 3)))
      data=[]
      #p = r"E:\Documents\My Projects\Instagram Dashboard\Model Development\flower_features.pkl"

      # lop through each image in the dataset
      for x in range(0,len(x_train_subclass)):
      # try to extract the features and update the dictionary
      #try:
          #print(x.shape)
          feat = extract_features(x_train_subclass[x],model)
          #data.append(x)
          data.append(feat)
      # if something fails, save the extracted features as a pickle file (optional)
      #except:
      #    with open(p,'wb') as file:
      #        pickle.dump(data,file)
          
 
      # get a list of the filenames
      #filenames = np.array(list(data.keys()))


      print(len(data))
      data = np.array(data)


      print(data.shape)

      # get a list of just the features
      #feat = data.reshape(-1,3072)
      # reshape so that there are 210 samples of 4096 vectors
      #feat = feat.reshape(-1,3072)
      # get the unique labels (from the flower_labels.csv)
      #df = pd.read_csv('flower_labels.csv')
      #label = df['label'].tolist()
      #unique_labels = list(set(label))
    
      
      data = data.reshape(-1,512)
      pca = PCA(n_components=100, random_state=22)
      pca.fit(data)
      x = pca.transform(data)
      #kmeans = KMeans(n_clusters=3,random_state=22)
      #kmeans.fit(x)
      #labels=kmeans.labels_
      #A_sparse = sparse.csr_matrix(x)
      #similarities = cosine_similarity(A_sparse)
      #clustering = SpectralClustering(n_clusters=5,  assign_labels='discretize',affinity='nearest_neighbors',  random_state=22).fit(similarities)
      #labels = clustering.labels_
      return x


# In[144]:





# In[145]:


data.shape


# In[146]:


featured_subclass= data.flatten()
label = np.array(featured_subclass)

dataset = pd.DataFrame({'label': label})


# In[149]:


x_train_subclass,y_train_subclass,x = find_classes(0)
reducer2 = UMAP(n_neighbors=15, n_components=2, n_epochs=1000, 
                min_dist=0.1, local_connectivity=2, random_state=42,
              )
featured_subclass = reducer2.fit_transform(x_train_subclass)
import pandas as pd
import numpy as np
#images = np.array(featured_subclass)
featured_subclass= featured_subclass.flatten()
label = np.array(featured_subclass)

dataset = pd.DataFrame({'label': label})


# In[43]:





# In[34]:





# In[169]:


x_train_subclass,y_train_subclass,x = find_classes(9)
data=kmeans_clustering(x_train_subclass,y_train_subclass,x)
featured_subclass= data.flatten()
label = np.array(featured_subclass)

dataset = pd.DataFrame({'label': label})


# In[170]:


Sum_of_squared_distances = []
K = range(1,10)
for num_clusters in K :
 kmeans = KMeans(n_clusters=num_clusters)
 kmeans.fit(dataset)
 Sum_of_squared_distances.append(kmeans.inertia_)
plt.plot(K,Sum_of_squared_distances,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Sum of squared distances/Inertia') 
plt.title('Elbow Method For Optimal k')
plt.show()


# In[148]:


range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
silhouette_avg = []
for num_clusters in range_n_clusters:
 
 # initialise kmeans
 kmeans = KMeans(n_clusters=num_clusters)
 kmeans.fit(dataset)
 cluster_labels = kmeans.labels_
 
 # silhouette score
 silhouette_avg.append(silhouette_score(dataset, cluster_labels))
plt.plot(range_n_clusters,silhouette_avg,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Silhouette score') 
plt.title('Silhouette analysis For Optimal k')
plt.show()


# In[76]:


####reducer umap
kmeans =[3,3,2,3,3,3,3,3,3,3]
silhhioutte = [2,3,2,3,2,2,2,2,2,4]


# In[ ]:


###Vgg16-transfer feature and kmeans cluster
Kmeans=[4,5,4,6,4,4,4,5,4,4]
silhhioutte=[]


# In[175]:


import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam


# In[178]:


from __future__ import print_function
import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers

class cifar100vgg:
    def __init__(self,train=True):
        self.num_classes = 100
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]

        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights('cifar100vgg.h5')


    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        print(mean)
        print(std)
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 121.936
        std = 68.389
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size)

    def train(self,model):

        #training parameters
        batch_size = 128
        maxepoches = 250
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20

        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)


        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)


        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)



        #optimization details
        sgd = SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


        # training process in a for loop with learning rate drop every 25 epoches.

        historytemp = model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=maxepoches,
                            validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=1)
        model.save_weights('cifar100vgg.h5')
        return model

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = tf.keras.utils.to_categorical(y_train, 100)
    y_test = tf.keras.utils.to_categorical(y_test, 100)

    model = cifar100vgg()

    predicted_x = model.predict(x_test)
    residuals = (np.argmax(predicted_x,1)!=np.argmax(y_test,1))
    loss = sum(residuals)/len(residuals)
    print("the validation 0/1 loss is: ",loss)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[268]:





# In[ ]:





# In[219]:





# In[220]:


from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Conv2DTranspose
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Input, Dense, Reshape, Conv2DTranspose,   Activation, BatchNormalization, ReLU, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import cifar100, cifar10


# In[221]:


def conv_block(x, filters, kernel_size, strides=2):
   x = Conv2D(filters=filters,
              kernel_size=kernel_size,
              strides=strides,
              padding='same')(x)
   x = BatchNormalization()(x)
   x = ReLU()(x)
   return x
def deconv_block(x, filters, kernel_size):
   x = Conv2DTranspose(filters=filters,
                       kernel_size=kernel_size,
                       strides=2,
                       padding='same')(x)
   x = BatchNormalization()(x)
   x = ReLU()(x)
   return x


# In[239]:


def denoising_autoencoder():
   dae_inputs = Input(shape=(32, 32, 3), name='dae_input')
   conv_block1 = conv_block(dae_inputs, 32, 3)
   conv_block2 = conv_block(conv_block1, 64, 3)
   conv_block3 = conv_block(conv_block2, 128, 3)
   conv_block4 = conv_block(conv_block3, 256, 3)
   conv_block5 = conv_block(conv_block4, 256, 3)
   conv_block6 = conv_block(conv_block5, 256, 3, 1)

   deconv_block0= deconv_block(conv_block6 ,256, 3)
   merge0 = Concatenate()([deconv_block0,  conv_block4])
   deconv_block1 = deconv_block(merge0, 256, 3)
   merge1 = Concatenate()([deconv_block1, conv_block3])
   deconv_block2 = deconv_block(merge1, 128, 3)
   merge2 = Concatenate()([deconv_block2, conv_block2])
   deconv_block3 = deconv_block(merge2, 64, 3)
   merge3 = Concatenate()([deconv_block3, conv_block1])
   deconv_block4 = deconv_block(merge3, 32, 3)

   final_deconv = Conv2DTranspose(filters=3,
                       kernel_size=3,
                       padding='same')(deconv_block4)
   ##change
   #final_deconv1= Dropout(0.25)(final_deconv)

   dae_outputs = Activation('sigmoid', name='dae_output')(final_deconv)
  
   return Model(dae_inputs, dae_outputs, name='dae')


# In[240]:


dae = denoising_autoencoder()
dae.compile(loss='huber_loss', optimizer='adam')


# In[241]:


dae.summary()


# In[ ]:





# In[ ]:





# In[ ]:





# In[254]:





# In[ ]:




        


# In[ ]:





# In[265]:


38//10


# In[274]:


cifar_x_test.shape


# In[271]:


softmax = tf.nn.softmax([-1, 0., 1.])


# In[272]:


softmax


# In[283]:


prediction = tf.nn.softmax(cifar_x_test[100:101].reshape(-1,3072))


# In[287]:


x=sum(prediction)


# In[289]:


len(x)


# In[291]:


tf.argmax(cifar_x_test[100:101].reshape(32,32,3), 1)


# In[280]:


f


# In[281]:


softmax = tf.nn.softmax([-1, 0., 1.])


# In[282]:


softmax


# In[ ]:





# In[293]:


loaded_graph = tf.Graph()


# In[296]:


import numpy as np

x = [1, 2, 3]
y = [4, 5, 6]


# In[297]:


print(x,y)


# In[298]:


zipped = zip(x, y)


# In[299]:


print(zipped)


# In[ ]:





# In[ ]:





# In[ ]:





# In[319]:





# In[ ]:





# In[ ]:





# In[325]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[347]:





# In[ ]:





# In[ ]:





# In[378]:





# In[380]:





# In[ ]:




