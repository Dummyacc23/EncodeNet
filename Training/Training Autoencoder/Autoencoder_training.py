#!/usr/bin/env python
# coding: utf-8

# In[18]:



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


# In[19]:


import tensorflow as tf
print(tf.__version__)
from keras.applications.vgg16 import preprocess_input


# In[20]:


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
    data_dir ="B_net/datasets/data/cifar10"
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


# In[21]:


print(train_labels)


# In[22]:


plt.imshow(cifar_x_test[4000:4001].reshape(32,32,3))


# In[5]:


x_train, train_filenames, y_train, x_test, test_filenames, y_test, label_names =         load_cifar_10_data(data_dir)


# In[6]:


print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


# In[7]:


fig, ax = plt.subplots(5, 5)
k = 0
 
for i in range(5):
    for j in range(5):
        ax[i][j].imshow(x_train[k], aspect='auto')
        k += 1
 
plt.show()


# In[8]:


def find_classes(x):
    x_train_subclass=[]
    y_train_subclass=[]
    index = np.where(y_train == x)
    #print(index[0])
    for indices in index[0]:
       x_train_subclass.append(x_train[indices])
       y_train_subclass.append(y_train[indices])
    #print(len(x_train_subclass))
    #plt.imshow(x_train_subclass[10].reshape(32,32,3))
    x_train_subclass = np.array(x_train_subclass)
    x_train_subclass= x_train_subclass.reshape(5000,3072)
    print(len(x_train_subclass), len(y_train_subclass))
    
    return x_train_subclass,y_train_subclass,x


# In[9]:


find_classes(0)


# In[10]:


reducer2 = UMAP(n_neighbors=15, n_components=2, n_epochs=1000, 
                min_dist=0.1, local_connectivity=2, random_state=42,
              )
def Hdb_cluster(x_train_subclass,y_train_subclass,x):
   featured_subclass = reducer2.fit_transform(x_train_subclass)
   print(featured_subclass.shape)
   clusterer = hdbscan.HDBSCAN(min_cluster_size=100)
   cluster_labels = clusterer.fit_predict(featured_subclass) 
   print(clusterer.labels_)
   clusterer.labels_.max()
   return cluster_labels
   
   
 


# In[279]:


import numpy as np
import pandas as pd 
def train_exits():
    df = pd.read_csv('Y_train.txt', header= None, usecols=[0], sep='\t')
    train_exits= df.values.tolist()
    train_exits= np.array(train_exits)
    train_exits=train_exits.reshape(50000,1)
    return train_exits


# In[113]:


train_exits= train_exits()


# In[14]:


cifar_x_train=x_train
cifar_x_test =x_test
cifar_y_train = y_train
cifar_y_test = y_test
cifar_x_train, cifar_x_test = x_train / 255.0, x_test / 255.0
 
# flatten the label values
cifar_y_train, cifar_y_test = y_train.flatten(), y_test.flatten()


# In[15]:


x_t,t_t, x = find_classes(0)
labels = Hdb_cluster(x_t,t_t, x)


# In[60]:


print(len(labels))


# In[16]:


groups = {}
for file, cluster in zip(x_t,labels):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)


# In[71]:


#print(len(set(labels)))
cluster_list=[]
for idx in set(labels):
    #print(idx)
    cluster_indexes = "cluster_indexes"
    cluster_indexes= cluster_indexes+str(idx)
    #cluster_indexes= np.where(labels == idx)
    print(cluster_indexes)
    cluster_list.append(cluster_indexes)
    


# In[72]:


print(len(cluster_list))


# In[85]:


real_index = np.where(y_train == 0)


# In[102]:


''''total_hard=[]
total_easy=[]

for idx,clusters in zip(set(labels),cluster_list):
    print(clusters)
    print(idx)
    clusters= np.where(labels == idx)
    print(len(clusters[0]))
    #print(clusters[0])
    print(len(clusters[0]))
    clusters_realindexes=[]
    for x in clusters[0]:
        clusters_realindexes.append(real_index[0][x])   
    print(len(clusters_realindexes))
    cluster_easyindex=[]
    cluster_hardindex=[]
    cluster_easyimages=[]
    cluster_hardimages=[]
    for idx in clusters_realindexes:
        if(train_exits[idx]==0):
           cluster_easyindex.append(idx)
           cluster_easyimages.append(x_train[idx])
        else:
           cluster_hardindex.append(idx)
           cluster_hardimages.append(x_train[idx])
    print(len(cluster_easyimages))
    if((len(cluster_easyimages)>=(len(cluster_hardimages)))):
        total_hard.append(cluster_hardimages)
        total_easy.append(cluster_easyimages[:len(cluster_hardimages)])
    else:
        total_hard.append(cluster_hardimages[:len(cluster_easyimages)])
        total_easy.append(cluster_easyimages)
        
''''''


# In[114]:


len(total_hard[2])


# In[115]:


len(total_easy[2])


# In[23]:


def calculate_entropy(model,cifar_x_test):
    cifar_x_test=np.array(cifar_x_test)
    xdata = cifar_x_test.reshape(-1, 32, 32, 3)
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


# In[12]:


print(calculate_entropy(model,cifar_x_test))


# In[281]:


score = model.evaluate( cifar_x_test, cifar_y_test, batch_size=32, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[720]:


overall_acc= []
for idx in range(0,len(cifar_x_train)):
    score = model.evaluate( cifar_x_train[idx:idx+1], cifar_y_train[idx:idx+1], batch_size=32, verbose=1)
    overall_acc.append(score[1])
    


# In[ ]:





# In[162]:


if(overall_acc[1000]==1):
    print("true")
else:
    print("False")


# In[1]:


get_ipython().system('pip3 install chainer')


# In[22]:


import chainer
import chainer.functions as F
from scipy.stats import entropy


# In[29]:


softmax = F.softmax(cifar_x_train,axis=0)
print(softmax.data)
cifar_x_train=np.array(cifar_x_train)
xdata = cifar_x_train.reshape(-1, 32, 32, 3)
#ydata = cifar_y_test
#softmax_data = model.predict(xdata)
entropy_value = np.array([entropy(s) for s in softmax.data])
train_entroexits=[]
easy_entropy_data=[]
easy_entropy_label=[]

hard_entropy_data=[]
hard_entropy_level=[]


# In[24]:


print(entropy_value[1000])


# In[53]:


import dill

with open("models/cifar10model.bn", "wb") as f:
    dill.dump(model, f)


# In[24]:


import dill

with open("models/cifar10model.bn", "rb") as f:
    model = dill.load(f)


# In[331]:


import dill


# In[25]:


x_train_branchy=np.array(x_train_branchy)
xdata = x_train_branchy.reshape(-1, 32, 32, 3)
#ydata = cifar_y_test
softmax_data = model.predict(xdata)
#print(softmax_data)
entropy_value = np.array([entropy(s) for s in softmax_data])
train_entroexits_test=[]
easy_entropy_data_test=[]
easy_entropy_label_test=[]

hard_entropy_data_test=[]
hard_entropy_level_test=[]




def calculate_thresholded_entropy_value(thresholds):
    count =0
    count_hard=0
    test_set=[]
    test_ylabel=[]
    autoencode_set=[]
    autoencode_label=[]
    train_entroexits=[]
    #train_entroexits=[]
    
    for idx in range(0,len(entropy_value)):
        print(entropy_value[idx])
        if(entropy_value[idx]<thresholds):
            count+=1
            train_entroexits.append(0)
            #easy_entropy_data_test.append(cifar_x_test[idx:idx+1])
            #easy_entropy_label_test.append(cifar_y_test[idx:idx+1])
            
        else:
            count_hard+=1
            train_entroexits.append(1)
            #hard_entropy_data_test.append(cifar_x_test[idx:idx+1])
            #hard_entropy_level_test.append(cifar_y_test[idx:idx+1])
            
        
        
    print("Last")        
    print(count)
    print("easy data",count)
    print("Hard Data",count_hard)
    
    return train_entroexits


# In[130]:


train_entroexits= calculate_thresholded_entropy_value(0.1220909090909091)

###best threshold til now 0.1220909090909091


# In[282]:


len(train_entroexits)


# In[283]:


if(train_entroexits[5000]==0):
    print("True")
else:
    print("false")


# In[396]:


easy_entropy_data_test=np.array(easy_entropy_data_test).reshape(-1,32,32,3)
hard_entropy_data_test=np.array(hard_entropy_data_test).reshape(-1,32,32,3)
easy_entropy_label_test=np.array(easy_entropy_label_test)
easy_entropy_label_test= easy_entropy_label_test.flatten()
hard_entropy_level_test= np.array(hard_entropy_level_test)
hard_entropy_level_test= hard_entropy_level_test.flatten()


score = model.evaluate( easy_entropy_data_test, easy_entropy_label_test, batch_size=32, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[395]:


score = model.evaluate( hard_entropy_data_test, hard_entropy_level_test, batch_size=32, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[448]:


score = model.evaluate( cifar_x_test, cifar_y_test, batch_size=32, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[336]:


check= dae.predict(hard_entropy_data_test)
check=np.array(check)
check = check.reshape(-1,32,32,3)



# In[341]:


plt.figure(figsize=(20,20))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(hard_entropy_data_test[i].reshape(32,32,3), cmap=plt.cm.binary)
    #plt.xlabel(class_names[train_labels[i]])
plt.show()


# In[ ]:





# In[342]:


plt.figure(figsize=(10,10))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(check[i].reshape(32,32,3), cmap=plt.cm.binary)
    #plt.xlabel(class_names[train_labels[i]])
plt.show()


# In[337]:


score = model.evaluate( check, hard_entropy_level_test, batch_size=32, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


from sklearn.cluster import SpectralClustering
import numpy as np


# In[ ]:





# In[41]:


import random
'''
def clustering_bucketing(x_t,labels,x):
    groups = {}
    for file, cluster in zip(x_t,labels):
        if cluster not in groups.keys():
          groups[cluster] = []
          groups[cluster].append(file)
        else:
          groups[cluster].append(file)
    cluster_list=[]
    for idx in set(labels):
        #print(idx)
        cluster_indexes = "cluster_indexes"
        cluster_indexes= cluster_indexes+str(idx)
        #cluster_indexes= np.where(labels == idx)
        print(cluster_indexes)
        cluster_list.append(cluster_indexes)
   
    print(len(cluster_list))
    
    real_index = np.where(y_train == x)
    total_hard=[]
    total_easy=[]
    total_real_easy=[]

    for idx,clusters in zip(set(labels),cluster_list):
        print(clusters)
        print(idx)
        clusters= np.where(labels == idx)
        print(len(clusters[0]))
        #print(clusters[0])
        print(len(clusters[0]))
        clusters_realindexes=[]
        for x in clusters[0]:
           clusters_realindexes.append(real_index[0][x])   
        print(len(clusters_realindexes))
        cluster_easyindex=[]
        cluster_hardindex=[]
        cluster_easyimages=[]
        cluster_hardimages=[]
        #cluster_real_images=[]
        ###entropy is
        ###accuarcy ==1
        
        for idx in clusters_realindexes:
            if(train_exits[idx]==0):
            #if(train_entro_exits[idx]==0 and overall_acc[idx]==1):
            #if(train_entro_exits[idx]==0):
               cluster_easyindex.append(idx)
               cluster_easyimages.append(cifar_x_train[idx])
            else:
               cluster_hardindex.append(idx)
               cluster_hardimages.append(cifar_x_train[idx])
        print("Length of easy images in cluster",len(cluster_easyimages))
        print("Length of hard images in cluster",len(cluster_hardimages))
        
        
        if((len(cluster_easyimages)>=(len(cluster_hardimages)))):
             var_easy=[]
             total_hard.append(cluster_hardimages)
             #total_hard.append(cluster_hardimages)
             #total_hard.append(cluster_hardimages)
             #r = random.randint(0, len(cluster_hardimages))
             r = calculate_entropy(model,cluster_easyimages)
             print("entropy index",r)
             var_easy.append(cluster_easyimages[r])
             #print("random number",r)
             #print(var_easy.shape)
             easy_peasy1 = var_easy*len(cluster_hardimages)
             print("Actual length")
             print(len(cluster_hardimages))
             print("Easy length")
             print(len(easy_peasy1))
             #total_easy.append(cluster_easyimages[:len(cluster_hardimages)])
             total_easy.append(easy_peasy1)
             #total_easy.append(easy_peasy1)
             #total_easy.append(easy_peasy1)
             total_real_easy.append(cluster_easyimages[:len(cluster_hardimages)])
        else:
             
             total_hard.append(cluster_hardimages[:len(cluster_easyimages)])
             #total_hard.append(cluster_hardimages[:len(cluster_easyimages)])
             #total_hard.append(cluster_hardimages[:len(cluster_easyimages)])
             #total_easy.append(cluster_easyimages)
             var_easy=[]
             r = calculate_entropy(model,cluster_easyimages)
             print("entropy index",r)
             var_easy.append(cluster_easyimages[r])
             easy_peasy1 = var_easy*len(cluster_easyimages)
             print("Actual length")
             print(len(cluster_easyimages))
             print("Easy length")
             print(len(easy_peasy1))
             total_easy.append(easy_peasy1)
             #total_easy.append(easy_peasy1)
             #total_easy.append(easy_peasy1)
             total_real_easy.append(cluster_easyimages)
             
             
             
            
    
    
    return total_easy, total_hard,total_real_easy


# In[284]:


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


# In[ ]:





# In[132]:


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
      ##change 
      pca = PCA(n_components=100, random_state=22)
      pca.fit(data)
      x = pca.transform(data)
      #x = reducer2.fit_transform(data)
      kmeans = KMeans(n_clusters=5,random_state=22)
      kmeans.fit(x)
      labels=kmeans.labels_
      return labels


# In[783]:


len(train_entroexits)


# In[133]:


def clustering_bucketing(x_t,labels,x):
    groups = {}
    for file, cluster in zip(x_t,labels):
        if cluster not in groups.keys():
          groups[cluster] = []
          groups[cluster].append(file)
        else:
          groups[cluster].append(file)
    cluster_list=[]
    for idx in set(labels):
        #print(idx)
        cluster_indexes = "cluster_indexes"
        cluster_indexes= cluster_indexes+str(idx)
        #cluster_indexes= np.where(labels == idx)
        print(cluster_indexes)
        cluster_list.append(cluster_indexes)
   
    print(len(cluster_list))
    
    real_index = np.where(y_train == x)
    total_hard=[]
    total_easy=[]
    total_real_easy=[]

    for idx,clusters in zip(set(labels),cluster_list):
        print(clusters)
        print(idx)
        clusters= np.where(labels == idx)
        print(len(clusters[0]))
        #print(clusters[0])
        print(len(clusters[0]))
        clusters_realindexes=[]
        for x in clusters[0]:
           clusters_realindexes.append(real_index[0][x])   
        print(len(clusters_realindexes))
        cluster_easyindex=[]
        cluster_hardindex=[]
        cluster_easyimages=[]
        cluster_hardimages=[]
        #cluster_real_images=[]
        ###entropy is
        ###accuarcy ==1
        
        for idx in clusters_realindexes:
            #if(train_exits[idx]==0):
            #if(train_entro_exits[idx]==0 and overall_acc[idx]==1):
            #if(train_entro_exits[idx]==0):
            if(train_entroexits[idx]==0):
               cluster_easyindex.append(idx)
               cluster_easyimages.append(cifar_x_train[idx])
            else:
               cluster_hardindex.append(idx)
               cluster_hardimages.append(cifar_x_train[idx])
        print("Length of easy images in cluster",len(cluster_easyimages))
        print("Length of hard images in cluster",len(cluster_hardimages))
        
        
        if((len(cluster_easyimages)>=(len(cluster_hardimages)))):
             
             #total_hard.append(cluster_hardimages)
             ###begin changes
             d= len(cluster_easyimages)-len(cluster_hardimages)
             if(d==0):
                    var_easy=[]
                    total_hard.append(cluster_hardimages)
                    total_real_easy.append(cluster_easyimages)
                    #var_easy.append(cluster_easyimages[r])
                    #total_hard.append(cluster_hardimages)
                    #total_hard.append(cluster_hardimages)
                    #r = random.randint(0, len(cluster_hardimages))
                    r = calculate_entropy(model,cluster_easyimages)
                    #print("entropy index",r)
                    var_easy.append(cluster_easyimages[r])
                    #print("random number",r)
                    #print(var_easy.shape)
                    easy_peasy1 = var_easy*len(cluster_easyimages)
                    #print("Actual length")
                    #print(len(cluster_easyimages))
                    #print("Easy length")
                    #print(len(easy_peasy1))
                    #total_easy.append(cluster_easyimages[:len(cluster_hardimages)])
                    total_easy.append(easy_peasy1)
                    #total_easy.append(easy_peasy1)
                    #total_easy.append(easy_peasy1)
             else:
                    var_easy=[]
                    remaining=d//len(cluster_hardimages)
                    total_real_easy.append(cluster_easyimages)
                    total_hard.append(cluster_hardimages)
                    length1 = len(cluster_easyimages)
                    length2 = len(cluster_hardimages)
                    for itera in range(0,remaining+1):
                        left = length1 - length2
                        if(left<=length2):
                            total_hard.append(cluster_hardimages[:left])
                        else:
                            total_hard.append(cluster_hardimages[:length2])
                            length1=left
                    #total_hard.append(cluster_hardimages)
                    #total_hard.append(cluster_hardimages)
                    #r = random.randint(0, len(cluster_hardimages))
                    r = calculate_entropy(model,cluster_easyimages)
                    #print("entropy index",r)
                    var_easy.append(cluster_easyimages[r])
                    #print("random number",r)
                    #print(var_easy.shape)
                    easy_peasy1 = var_easy*len(cluster_easyimages)
                    #print("Actual length")
                    #print(len(cluster_easyimages))
                    #print("Easy length")
                    #print(len(easy_peasy1))
                    #total_easy.append(cluster_easyimages[:len(cluster_hardimages)])
                    total_easy.append(easy_peasy1)
                    #total_easy.append(easy_peasy1)
                    #total_easy.append(easy_peasy1)
                    
                            
                            
                    
                
                    
                    
             
             #total_real_easy.append(cluster_easyimages[:len(cluster_hardimages)])
        else:
             '''
             total_hard.append(cluster_hardimages[:len(cluster_easyimages)])
             #total_hard.append(cluster_hardimages[:len(cluster_easyimages)])
             #total_hard.append(cluster_hardimages[:len(cluster_easyimages)])
             #total_easy.append(cluster_easyimages)
             var_easy=[]
             r = calculate_entropy(model,cluster_easyimages)
             print("entropy index",r)
             var_easy.append(cluster_easyimages[r])
             easy_peasy1 = var_easy*len(cluster_easyimages)
             print("Actual length")
             print(len(cluster_easyimages))
             print("Easy length")
             print(len(easy_peasy1))
             total_easy.append(easy_peasy1)
             #total_easy.append(easy_peasy1)
             #total_easy.append(easy_peasy1)
             total_real_easy.append(cluster_easyimages)
             '''
             var_easy=[]
             d= len(cluster_hardimages)-len(cluster_easyimages)
             remaining=d//len(cluster_easyimages)
             total_real_easy.append(cluster_easyimages)
             total_hard.append(cluster_hardimages)
             length1 = len(cluster_hardimages)
             length2 = len(cluster_easyimages)
             for itera in range(0,remaining+1):
                left = length1 - length2
                print("Left item",left)
                
                if(left<=length2):
                    total_real_easy.append(cluster_easyimages[:left])
                else:
                    total_real_easy.append(cluster_easyimages[:length2])
                    length1=left
                    #print("loop",length1)
                            
             #r = calculate_entropy(model,cluster_easyimages)
             #print("entropy index",r)
             #var_easy.append(cluster_easyimages[r])
             #total_hard.append(cluster_hardimages)
             #total_hard.append(cluster_hardimages)
             #r = random.randint(0, len(cluster_hardimages))
             r = calculate_entropy(model,cluster_easyimages)
             print("entropy index",r)
             var_easy.append(cluster_easyimages[r])
             #print("random number",r)
             #print(var_easy.shape)
             easy_peasy1 = var_easy*len(cluster_hardimages)
             #print("Actual length 2nd condition")
             print(len(cluster_hardimages))
             #print("Easy length 2nd condition")
             print(len(easy_peasy1))
             #total_easy.append(cluster_easyimages[:len(cluster_hardimages)])
             total_easy.append(easy_peasy1)
             #total_easy.append(easy_peasy1)
             #total_easy.append(easy_peasy1)
                    
             
             
             
            
    
    
    return total_easy, total_hard,total_real_easy
    


# In[66]:


#totaL_ea,total_ha= clustering_bucketing(0)


# In[1]:


x_t,t_t, x = find_classes(0)
labels = Hdb_cluster(x_t,t_t, x)


# In[131]:


print(x)


# In[796]:


totaL_ea,total_ha,total_re=clustering_bucketing(x_t,labels,x)


# In[805]:


count =0
for idx in range(0,len(total_re)):
    count+=len(total_re[idx])


# In[449]:


count 


# In[134]:


def find_minimum(a,b,c):
    smallest=0
    print(a,b,c)

    if a <= b and a <= c :
        smallest = a
    elif b <= a and b <= c :
        smallest = b
    elif c <=a and c <= b :
        smallest = c
    return smallest


# In[122]:


from keras.applications.vgg16 import preprocess_input 


# In[450]:


total_easy_all=[]
total_hard_all=[]
val_easy_all=[]
val_hard_all=[]
total_reasy_all=[]
val_reasy_all=[]

for final_class in range(0,10):
    x_t,t_t, x = find_classes(final_class)
    #labels = Hdb_cluster(x_t,t_t, x)
    labels = kmeans_clustering(x_t,t_t, x)
    print(x)
    totaL_ea,total_ha,total_reasy=clustering_bucketing(x_t,labels,x)
    print("total_easy_subclass",len(totaL_ea))
    print("total_hard_subclass",len(total_ha))
    print("total_realeasy_subclass",len(total_reasy))
    
    #print(len(totaL_ea),len(total_ha))
    for idx in range(0,len(totaL_ea)):
        total_easy_all+=totaL_ea[idx]
        val_easy_all+=totaL_ea[idx][:(len(totaL_ea[idx]))//3]
    
    for idx in range(0,len(total_ha)):
        total_hard_all+=(total_ha[idx])
        val_hard_all+=total_ha[idx][:(len(total_ha[idx]))//3]
    for idx in range(0,len(total_reasy)):
        total_reasy_all+=(total_reasy[idx])
        val_reasy_all+=total_reasy[idx][:(len(total_reasy[idx]))//3]
        
           

            


# In[451]:


print(len(total_easy_all))
print(len(total_hard_all))
print(len(total_reasy_all))


# In[29]:


print(len(val_easy_all))
print(len(val_hard_all))


# In[453]:


print(len(total_reasy_all))
print(len(val_reasy_all))


# In[351]:


num= find_minimum(len(val_easy_all),len(val_hard_all),len(val_reasy_all))
print(num)


# In[287]:


num= find_minimum(len(val_easy_all),len(val_hard_all),len(val_reasy_all))
num1 = find_minimum(len(total_easy_all),len(total_hard_all),len(total_reasy_all))
print(num)
val_easy_all=val_easy_all[:num]
val_hard_all=val_hard_all[:num]
val_reasy_all=val_reasy_all[:num]
total_easy_all=total_easy_all[:num1]
total_hard_all=total_hard_all[:num1]
total_reasy_all=total_reasy_all[:num1]


# In[288]:


#####previous best for 75% accuracy
total_hard= total_hard_all+total_reasy_all+total_hard_all+total_reasy_all
total_easy= total_easy_all+total_easy_all+total_easy_all+total_easy_all
val_hard= val_hard_all+val_reasy_all+val_hard_all+val_reasy_all
val_easy= val_easy_all+val_easy_all+val_easy_all+val_easy_all

######################################


# In[196]:


total_hard= total_hard_all+total_reasy_all+total_hard_all+total_reasy_all+total_hard_all+total_reasy_all
total_easy= total_easy_all+total_easy_all+total_easy_all+total_easy_all+total_easy_all+total_easy_all
val_hard= val_hard_all+val_reasy_all+val_hard_all+val_reasy_all+val_hard_all+val_reasy_all
val_easy= val_easy_all+val_easy_all+val_easy_all+val_easy_all+val_easy_all+val_easy_all


# In[ ]:





# In[752]:


total_hard= total_reasy_all+total_hard_all+total_reasy_all+total_hard_all
total_easy= total_easy_all+total_easy_all+total_easy_all+total_easy_all
val_hard= val_reasy_all+val_hard_all+val_reasy_all+val_hard_all
val_easy= val_easy_all+val_easy_all+val_easy_all+val_easy_all


# In[290]:


total_hard=np.array(total_hard)
total_easy=np.array(total_easy)
val_hard=np.array(val_hard)
val_easy=np.array(val_easy)


# In[291]:


total_hard=total_hard.reshape(-1,32,32,3)
total_easy=total_easy.reshape(-1,32,32,3)
val_hard=val_hard.reshape(-1,32,32,3)
val_easy=val_easy.reshape(-1,32,32,3)

print(total_hard.shape)
print(total_easy.shape)
print(val_hard.shape)
print(val_easy.shape)


# In[292]:


it_train = datagen.flow(total_hard, total_easy, batch_size=64)


# In[ ]:





# In[259]:


plt.imshow(cifar_x_train[5008].reshape(32,32,3))


# In[260]:


plt.imshow(x_train[5008].reshape(32,32,3))


# In[216]:


plt.imshow(total_easy[5008].reshape(32,32,3))


# In[ ]:


plt.figure(figsize=(20,20))
count = 101
for i in range(0,100,1):
    plt.subplot(10,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(total_hard_all[i].reshape(32,32,3), cmap=plt.cm.binary)
    #plt.xlabel(class_names[train_labels[i]])
    #count = count+1
plt.show()


# In[ ]:


plt.figure(figsize=(20,20))
count = 101
for i in range(0,100,1):
    plt.subplot(10,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(total_reasy_all[i].reshape(32,32,3), cmap=plt.cm.binary)
    #plt.xlabel(class_names[train_labels[i]])
    #count = count+1
plt.show()


# In[ ]:




for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[88]:


print(len(val_easy_all))
print(len(val_hard_all))


# In[67]:


from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation

from tensorflow.keras import regularizers

input_img = Input(shape=(32, 32,3))
x = Conv2D(32, (3, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), padding='same')(encoded)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(3, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
decoded = Activation('sigmoid')(x)

autoencoder = tf.keras.Model(input_img, decoded)


# In[68]:


total_hard_all=np.array(total_hard_all.reshape(-1,32,32,3))
total_easy_all= np.array(total_easy_all.reshape(-1,32,32,3))
val_hard_all= np.array(val_hard_all.reshape(-1,32,32,3))
val_easy_all= np.array(val_easy_all.reshape(-1,32,32,3))


# In[69]:


total_hard_all=np.array(total_hard_all)
total_hard_all= total_hard_all/255.0
total_easy_all= np.array(total_easy_all)
total_easy_all= total_easy_all/255.0
val_hard_all= np.array(val_hard_all)
val_hard_all=val_hard_all/255.0
val_easy_all=np.array(val_easy_all)
val_easy_all= val_easy_all/255.0


# In[70]:


total_hard_all=np.array(total_hard_all.reshape(-1,32,32,3))
total_easy_all= np.array(total_easy_all.reshape(-1,32,32,3))
val_hard_all= np.array(val_hard_all.reshape(-1,32,32,3))
val_easy_all= np.array(val_easy_all.reshape(-1,32,32,3))


# In[23]:


total_hard_all=np.array(total_hard_all/255.0)
total_easy_all=total_easy_all/255.0
val_hard_all=val_hard_all/255.0
val_easy_all= val_easy_all/255.0


# In[149]:


total_hard_all=np.array(total_hard_all)
total_easy_all= np.array(total_easy_all)
val_hard_all= np.array(val_hard_all)
val_easy_all=np.array(val_easy_all)


# In[150]:


total_hard_all=total_hard_all.reshape(-1,32,32,3)
total_easy_all= total_easy_all.reshape(-1,32,32,3)
val_hard_all= val_hard_all.reshape(-1,32,32,3)
val_easy_all= val_easy_all.reshape(-1,32,32,3)


# In[192]:



 
plt.imshow(total_easy_all[5200].reshape(32,32,3))
    
    
    


# In[191]:


plt.imshow(total_hard_all[5200].reshape(32,32,3))


# In[178]:


import random


# In[26]:


plt.imshow(total_hard_all[5200].reshape(32,32,3))


# In[107]:


plt.imshow((total_easy_all[502].reshape(32,32,3)))


# In[124]:


plt.imshow((total_hard_all[501].reshape(32,32,3)))


# In[113]:


check = autoencoder.predict(total_easy_all[502].reshape(-1,32,32,3))


# In[114]:


plt.imshow(check.reshape(32,32,3))


# In[92]:



#####Autoencoder without normalization
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')

history = autoencoder.fit(total_easy_all.reshape(len(total_hard_all), 32, 32,3), total_hard_all.reshape(len(total_hard_all),32, 32,3),
                epochs=20,
                batch_size=256,
                shuffle=True,
                validation_data=(val_easy_all.reshape(len(val_easy_all), 32, 32,3), val_hard_all.reshape(len(val_hard_all),32, 32,3)))


# In[ ]:





# In[64]:


plt.imshow((total_hard_all[1000].reshape(32,32,3)))


# In[ ]:





# In[ ]:





# In[32]:


from keras.preprocessing.image import ImageDataGenerator


# In[33]:


datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)


# In[35]:


#it_train = datagen.flow(total_easy_all, total_hard_all, batch_size=64)


# In[ ]:





# In[123]:


import numpy
def global_contrast_normalize(X, scale=1., subtract_mean=True, use_std=False,sqrt_bias=0., min_divisor=1e-8):
    """
    Global contrast normalizes by (optionally) subtracting the mean
    across features and then normalizes by either the vector norm
    or the standard deviation (across features, for each example).
    Parameters
    ----------
    X : ndarray, 2-dimensional
        Design matrix with examples indexed on the first axis and \
        features indexed on the second.
    scale : float, optional
        Multiply features by this const.
    subtract_mean : bool, optional
        Remove the mean across features/pixels before normalizing. \
        Defaults to `True`.
    use_std : bool, optional
        Normalize by the per-example standard deviation across features \
        instead of the vector norm. Defaults to `False`.
    sqrt_bias : float, optional
        Fudge factor added inside the square root. Defaults to 0.
    min_divisor : float, optional
        If the divisor for an example is less than this value, \
        do not apply it. Defaults to `1e-8`.
    Returns
    -------
    Xp : ndarray, 2-dimensional
        The contrast-normalized features.
    Notes
    -----
    `sqrt_bias` = 10 and `use_std = True` (and defaults for all other
    parameters) corresponds to the preprocessing used in [1].
    References
    ----------
    .. [1] A. Coates, H. Lee and A. Ng. "An Analysis of Single-Layer
       Networks in Unsupervised Feature Learning". AISTATS 14, 2011.
       http://www.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf
    """
    assert X.ndim == 2, "X.ndim must be 2"
    scale = float(scale)
    assert scale >= min_divisor

    # Note: this is per-example mean across pixels, not the
    # per-pixel mean across examples. So it is perfectly fine
    # to subtract this without worrying about whether the current
    # object is the train, valid, or test set.
    mean = X.mean(axis=1)
    if subtract_mean:
        X = X - mean[:, numpy.newaxis]  # Makes a copy.
    else:
        X = X.copy()

    if use_std:
        # ddof=1 simulates MATLAB's var() behaviour, which is what Adam
        # Coates' code does.
        ddof = 1

        # If we don't do this, X.var will return nan.
        if X.shape[1] == 1:
            ddof = 0

        normalizers = numpy.sqrt(sqrt_bias + X.var(axis=1, ddof=ddof)) / scale
    else:
        normalizers = numpy.sqrt(sqrt_bias + (X ** 2).sum(axis=1)) / scale

    # Don't normalize by anything too small.
    normalizers[normalizers < min_divisor] = 1.

    X /= normalizers[:, numpy.newaxis]  # Does not make a copy.
    return X


# In[ ]:





# In[ ]:


def normalize_data(x):
    data = x
    data = data/255.0
    mean = data.mean(axis=0)
    data -= mean
    ret = global_contrast_normalize(data)
    data = ret.reshape((-1, 3, 32, 32))
    return data


# In[175]:


total_hard_all_normalized=normalize_data(total_hard_all.reshape(-1,3072))
total_easy_all_normalized=normalize_data(total_easy_all.reshape(-1,3072))
val_hard_all_normalized=normalize_data(val_hard_all.reshape(-1,3072))
val_easy_all_normalized=normalize_data(val_easy_all.reshape(-1,3072))



# In[176]:


print(total_hard_all_normalized.shape)
print(total_easy_all_normalized.shape)
print(val_hard_all_normalized.shape)
print(val_easy_all_normalized.shape)




# In[177]:


def process_normalized(normalized_data):
    data=[]
    for x in range(0,len(normalized_data)):
        data.append((normalized_data[x]*255).reshape(32,32,3))
        
    return data


# In[178]:


total_hard_all_normalized = process_normalized(total_hard_all_normalized)
total_easy_all_normalized = process_normalized(total_easy_all_normalized)
val_hard_all_normalized = process_normalized(val_hard_all_normalized)
val_easy_all_normalized = process_normalized(val_easy_all_normalized)





# In[182]:


total_hard_all_normalized= np.array(total_hard_all_normalized)
total_easy_all_normalized= np.array(total_easy_all_normalized)
val_hard_all_normalized= np.array(val_hard_all_normalized)
val_easy_all_normalized= np.array(val_easy_all_normalized)


# In[171]:


data= np.array(data)


# In[172]:


data.shape


# In[179]:


plt.imshow((val_hard_all_normalized[1000]).reshape(32,32,3))


# In[183]:


autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')

history = autoencoder.fit(total_hard_all_normalized.reshape(len(total_hard_all_normalized), 32, 32, 3), total_easy_all_normalized.reshape(len(total_easy_all_normalized), 32, 32, 3),
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(val_hard_all_normalized.reshape(len(val_hard_all_normalized), 32, 32, 3), val_easy_all_normalized.reshape(len(val_easy_all_normalized), 32, 32, 3)))


# In[74]:


x_test.shape


# In[93]:


decoded_images = autoencoder.predict(cifar_x_test.reshape(cifar_x_test.shape[0],32,32,3))


# In[73]:


np.savetxt('cifar10_converted_test_data.txt', decoded_images.reshape(10000,3072))


# In[190]:


print(y_test[1000])


# In[ ]:





# In[43]:


import os
import sys
import glob
import numpy
import numpy as np
from six.moves import cPickle as pickle
from scipy import linalg
from skimage.color import rgb2luv
from skimage import img_as_float


# In[48]:


dirname = "B_net/"


# In[44]:


import numpy
def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()

    return data


# In[49]:


def global_contrast_normalize(X, scale=1., subtract_mean=True, use_std=False,
                              sqrt_bias=0., min_divisor=1e-8):
    """
    Global contrast normalizes by (optionally) subtracting the mean
    across features and then normalizes by either the vector norm
    or the standard deviation (across features, for each example).
    Parameters
    ----------
    X : ndarray, 2-dimensional
        Design matrix with examples indexed on the first axis and \
        features indexed on the second.
    scale : float, optional
        Multiply features by this const.
    subtract_mean : bool, optional
        Remove the mean across features/pixels before normalizing. \
        Defaults to `True`.
    use_std : bool, optional
        Normalize by the per-example standard deviation across features \
        instead of the vector norm. Defaults to `False`.
    sqrt_bias : float, optional
        Fudge factor added inside the square root. Defaults to 0.
    min_divisor : float, optional
        If the divisor for an example is less than this value, \
        do not apply it. Defaults to `1e-8`.
    Returns
    -------
    Xp : ndarray, 2-dimensional
        The contrast-normalized features.
    Notes
    -----
    `sqrt_bias` = 10 and `use_std = True` (and defaults for all other
    parameters) corresponds to the preprocessing used in [1].
    References
    ----------
    .. [1] A. Coates, H. Lee and A. Ng. "An Analysis of Single-Layer
       Networks in Unsupervised Feature Learning". AISTATS 14, 2011.
       http://www.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf
    """
    assert X.ndim == 2, "X.ndim must be 2"
    scale = float(scale)
    assert scale >= min_divisor

    # Note: this is per-example mean across pixels, not the
    # per-pixel mean across examples. So it is perfectly fine
    # to subtract this without worrying about whether the current
    # object is the train, valid, or test set.
    mean = X.mean(axis=1)
    if subtract_mean:
        X = X - mean[:, numpy.newaxis]  # Makes a copy.
    else:
        X = X.copy()

    if use_std:
        # ddof=1 simulates MATLAB's var() behaviour, which is what Adam
        # Coates' code does.
        ddof = 1

        # If we don't do this, X.var will return nan.
        if X.shape[1] == 1:
            ddof = 0

        normalizers = numpy.sqrt(sqrt_bias + X.var(axis=1, ddof=ddof)) / scale
    else:
        normalizers = numpy.sqrt(sqrt_bias + (X ** 2).sum(axis=1)) / scale

    # Don't normalize by anything too small.
    normalizers[normalizers < min_divisor] = 1.

    X /= normalizers[:, numpy.newaxis]  # Does not make a copy.
    return X


# In[46]:


import numpy
def get_data(gcn=1,whitening=1):
    data = np.zeros((50000, 3 * 32 * 32), dtype=np.float32)
    labels = np.zeros((50000), dtype=np.uint8)
    for i, data_fn in enumerate(
            sorted(glob.glob(dirname+'datasets/data/cifar10/data_batch*'))):
        batch = unpickle(data_fn)
        data[i * 10000:(i + 1) * 10000] = batch['data']
        labels[i * 10000:(i + 1) * 10000] = batch['labels']
    
    data /= 255
    mean = data.mean(axis=0)
    data -= mean
    
    if gcn==1:
        data = global_contrast_normalize(data,use_std=True)
        
    # if whitening == 1:
    #     components, meanw, data = preprocessing(data)

    data = data.reshape((-1, 3, 32, 32))
    # for i,image in enumerate(data):
    #     data[i] = rgb2luv(img_as_float(image).transpose((1,2,0))).transpose((2,0,1))


    # for i in range(50000):
    #     d = data[i]
    #     d -= d.min()
    #     d /= d.max()
    #     data[i] = d.astype(np.float32)
        
    train_data = data
    train_labels = np.asarray(labels, dtype=np.int32)

    test = unpickle(dirname+'datasets/data/cifar10/test_batch')
    data = np.asarray(test['data'], dtype=np.float32)
    
    data /= 255
    data -= mean

    if gcn==1:
        data = global_contrast_normalize(data,use_std=True)

    # if whitening == 1:
    #     mdata = data - meanw
    #     data = np.dot(mdata, components.T)

    data = data.reshape((-1, 3, 32, 32))
    # for i,image in enumerate(data):
    #     data[i] = rgb2luv(img_as_float(image).transpose((1,2,0))).transpose((2,0,1))


    # for i in range(10000):
    #     d = data[i]
    #     d -= d.min()
    #     d /= d.max()
    #     data[i] = d.astype(np.float32)
        
    test_data = data
    test_labels = np.asarray(test['labels'], dtype=np.int32)

    return train_data, train_labels, test_data, test_labels


# In[50]:


x_train_branchy,y_train_branchy,x_test_branchy, y_test_branchy = get_data()


# In[281]:


x_train_branchy.shape


# In[44]:


x_test_branchy.shape


# In[65]:


plt.imshow(x_test_branchy[1000].reshape(32,32,3))


# In[59]:


decoded_test= decoded_images


# In[60]:


plt.imshow(decoded_test[ ].reshape(32,32,3))


# In[ ]:





# In[11]:



import tensorflow as tf  
 
# Display the version
print(tf.__version__)    
 
# other imports
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model


# In[91]:


cifar_x_train=x_train
cifar_x_test =x_test
cifar_y_train = y_train
cifar_y_test = y_test


# In[92]:


cifar_x_train.shape


# In[93]:


cifar_x_train, cifar_x_test = x_train / 255.0, x_test / 255.0
 
# flatten the label values
cifar_y_train, cifar_y_test = y_train.flatten(), y_test.flatten()


# In[56]:


K = len(set(y_train))
 
# calculate total number of classes
# for output layer
print("number of classes:", K)
 
# Build the model using the functional API
# input layer
i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
 
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
 
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
 
x = Flatten()(x)
x = Dropout(0.2)(x)
 
# Hidden layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
 
# last hidden layer i.e.. output layer
x = Dense(K, activation='softmax')(x)
 
model = Model(i, x)
 
# model description
model.summary()


# In[57]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[58]:


r = model.fit(
  cifar_x_train, cifar_y_train, validation_data=(cifar_x_test, cifar_y_test), epochs=50)


# In[ ]:





# In[17]:


plt.plot(r.history['accuracy'], label='acc', color='red')
plt.plot(r.history['val_accuracy'], label='val_acc', color='green')
plt.legend()


# In[36]:


import os


# In[44]:


parent_path = os.getcwd()
print(parent_path)
directory= "data"


path = os.path.join(parent_path, directory)
print(path)


# In[47]:


if(os.path.exists(path)):
    print("True")
else:
    os.mkdir(path)
    print(path)


# In[54]:


path


# In[55]:


model.save('/home/hmahmud/.jupyter/branchynet/data/cifar10_tensorflow.model')


# In[56]:


model.load_weights('/home/hmahmud/.jupyter/branchynet/data/cifar10_tensorflow.model')


# In[57]:


batch_size = 32
score = model.evaluate(cifar_x_test, cifar_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[457]:


batch_size = 32
score = model.evaluate(cifar_x_test, cifar_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[94]:


decoded_images.shape


# In[95]:


autoencode_test = np.array(decoded_images)


# In[96]:


autoencode_test.shape


# In[97]:


autoencode_test= autoencode_test.reshape(10000,32,32,3)


# In[98]:


batch_size = 32
score = model.evaluate(autoencode_test, cifar_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[118]:


plt.imshow(autoencode_test[1000].reshape(32,32,3))


# In[27]:


plt.imshow(total_hard[1000].reshape(32,32,3))


# In[123]:


plt.imshow(cifar_x_test[1000].reshape(32,32,3))


# In[ ]:



    


# In[ ]:





# In[214]:


plt.imshow(total_easy_all[15000].reshape(32,32,3))


# In[213]:


plt.imshow(total_hard_all[15000].reshape(32,32,3))


# In[16]:


from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Conv2DTranspose


# In[15]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Input, Dense, Reshape, Conv2DTranspose,   Activation, BatchNormalization, ReLU, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import cifar100, cifar10


# In[48]:


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


def denoising_autoencoder():
   dae_inputs = Input(shape=(32, 32, 3), name='dae_input')
   conv_block1 = conv_block(dae_inputs, 32, 3)
   conv_block2 = conv_block(conv_block1, 64, 3)
   conv_block3 = conv_block(conv_block2, 128, 3)
   conv_block4 = conv_block(conv_block3, 256, 3)
   conv_block5 = conv_block(conv_block4, 256, 3, 1)

   deconv_block1 = deconv_block(conv_block5, 256, 3)
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


# In[49]:


def denoising_autoencoder():
   dae_inputs = Input(shape=(32, 32, 3), name='dae_input')
   conv_block1 = conv_block(dae_inputs, 32, 3)
   conv_block2 = conv_block(conv_block1, 64, 3)
   conv_block3 = conv_block(conv_block2, 128, 3)
   conv_block4 = conv_block(conv_block3, 256, 3)
   conv_block5 = conv_block(conv_block4, 256, 3, 1)

   deconv_block1 = deconv_block(conv_block5, 256, 3)
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

dae.summary()
# In[293]:


dae = denoising_autoencoder()
dae.compile(loss='mse', optimizer='adam')

#checkpoint = ModelCheckpoint('best_model.h5', verbose=1, save_best_only=True, save_weights_only=True)

history = dae.fit_generator(it_train,
                epochs=10,
                shuffle=True,
                validation_data=(val_hard.reshape(len(val_hard), 32, 32,3), val_easy.reshape(len(val_easy),32, 32,3)))


# In[247]:


import tensorflow.keras as K
import datetime
def decay(epoch):
    """ This method create the alpha"""
    # returning a very small constant learning rate
    return 0.001 / (1 + 1 * 30)


# In[650]:


import dill

with open("models/autoencoder_model_secondDisimilar.bn", "rb") as f:
    dae = dill.load(f)


# In[651]:


#score=dae.evaluate(x_test,y_test)
# we have changed the name of the layer from input_layer to conv_0
dae.layers[15]._name='encoded'  
#print('Test loss',score[0])
#print('Test accuracy',score[1])


# In[ ]:





# In[652]:


layer_output=dae.get_layer('encoded').output  #get the Output of the Layer
base_encoded=tf.keras.models.Model(inputs=dae.input,outputs=layer_output) #Intermediate model between Input Layer and Output Layer which we are concerned about
#intermediate_prediction=intermediate_model.predict(x_train[image_seq].reshape(1,28,28,1))


# In[654]:


Y_p = K.utils.to_categorical(cifar_y_train, 10)
Yv_p = K.utils.to_categorical(cifar_y_test, 10)


# In[251]:


cifar_x_train.shape


# In[257]:


encoder_model.summary()


# In[252]:


import tensorflow.keras as K


# In[661]:


temp_x_train=[]
temp_y_train=[]
temp_x_test=[]
temp_y_test=[]
similar_first =[0,1,2,3,7]
similar_2nd= [4,5,6,8,9]
similar_animals=[2,3,4,5,6,7]
similar_machines=[0,1,8,9]

for idx in range(0,len(cifar_x_train)):
    if cifar_y_train[idx:idx+1] in similar_2nd:
        temp_x_train.append(cifar_x_train[idx:idx+1])
        temp_y_train.append(cifar_y_train[idx:idx+1])
        
for idx in range(0,len(cifar_x_test)):
    if cifar_y_test[idx:idx+1] in similar_2nd:
        temp_x_test.append(cifar_x_test[idx:idx+1])
        temp_y_test.append(cifar_y_test[idx:idx+1])
        
temp_x_train= np.array(temp_x_train)
temp_x_train=temp_x_train.reshape(-1,32,32,3)
temp_y_train=np.array(temp_y_train)
temp_x_test= np.array(temp_x_test)
temp_x_test=temp_x_test.reshape(-1,32,32,3)
temp_y_test=np.array(temp_y_test)
#Chnage here
#temp_y_train = K.utils.to_categorical(temp_y_train, 10)
temp_y_train = K.utils.to_categorical(temp_y_train, 10)
temp_y_test = K.utils.to_categorical(temp_y_test, 10)


# In[658]:


temp_x_train.shape


# In[ ]:


###Lats try
Y_p = K.utils.to_categorical(cifar_y_train, 10)
Yv_p = K.utils.to_categorical(cifar_y_test, 10)


# In[664]:


encoder_model= K.Sequential()
# using upsamplign to get more data points and improve the predictions
#encoder_model.add(K.layers.UpSampling2D())
encoder_model.add(base_encoded)
encoder_model.add(K.layers.UpSampling2D())
encoder_model.add(K.layers.Flatten())
encoder_model.add(K.layers.Dense(128, activation=('relu')))
encoder_model.add(K.layers.Dropout(0.2))
encoder_model.add(K.layers.Dense(128, activation=('relu')))
encoder_model.add(K.layers.Dropout(0.2))
encoder_model.add(K.layers.Dense(10, activation=('softmax')))
# adding callbacks
callback = []
callback += [K.callbacks.LearningRateScheduler(decay, verbose=0)]
#callback += [K.callbacks.ModelCheckpoint('cifar10.h5',
#                                         save_best_only=True,
#                                        mode='min'
#                                         )]
# tensorboard callback
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# callback += [K.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]
# Compiling model with adam optimizer and looking the accuracy
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
it_train_alex = datagen.flow(temp_x_train, temp_y_train, batch_size=64)
encoder_model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
# training model with mini batch using shuffle data
rs=encoder_model.fit(it_train_alex,
batch_size=128,
validation_data=(temp_x_test, temp_y_test),
epochs=100, shuffle=True,
callbacks=callback,
verbose=1
)


# In[ ]:





# In[ ]:





# In[532]:



temp_x_test=[]
temp_y_test=[]
similar_first =[0,1,2,3,7]
similar_2nd= [4,5,6,8,9]
similar_animals=[2,3,4,5,6,7]
similar_machines=[0,1,8,9]


for idx in range(0,len(cifar_x_test)):
    if cifar_y_test[idx:idx+1] in similar_first:
        temp_x_test.append(cifar_x_test[idx:idx+1])
        temp_y_test.append(cifar_y_test[idx:idx+1])
        
temp_x_test= np.array(temp_x_test)
temp_x_test=temp_x_test.reshape(-1,32,32,3)
temp_y_test=np.array(temp_y_test)
        


# In[533]:


temp_y = K.utils.to_categorical(temp_y_test, 10)


# In[534]:


###300 epoch second dissimilar group 
batch_size = 32
score = model_first.evaluate(temp_x_test, temp_y, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[274]:


with open("models/firstgroup_classifier.bn", "rb") as f:
    model_first=dill.load(f)


# In[273]:



with open("models/secondgroup_classifier.bn", "rb") as f:
    model_second=dill.load(f)


# In[536]:


entropy_value_second[1000]


# In[548]:


Data=np.array(cifar_x_test)
xdata = Data.reshape(-1, 32, 32, 3)
#ydata = cifar_y_test
softmax_data1 = model_first.predict(xdata)
#print(softmax_data)
entropy_value_first = np.array([entropy(s) for s in softmax_data1])
softmax_data2 = model_second.predict(xdata)
#print(softmax_data)
entropy_value_second = np.array([entropy(s) for s in softmax_data2])

model_first_x_data=[]
model_first_y_data=[]
model_second_x_data=[]
model_second_y_data=[]
miss_predicted_autoencoder_first=[]
miss_predicted_autoencoder_second=[]
miss_predicted_autoencoder_first_index=[]
miss_predicted_autoencoder_second_index=[]




#entropy_value_first[idx]>entropy_value_second[idx] and 
#max_probability_first[idx]>max_probability_second[idx]

for idx in range(0, len(cifar_x_test)):
    if(entropy_value_first[idx]>=entropy_value_second[idx] and cifar_y_test[idx:idx+1] not in similar_2nd):
        miss_predicted_autoencoder_second.append(entropy_value_second[idx])
        miss_predicted_autoencoder_second_index.append(idx)
        miss_predicted_autoencoder_first.append(entropy_value_first[idx])
        
    if(entropy_value_first[idx]<entropy_value_second[idx] and cifar_y_test[idx:idx+1] not in similar_first):
        miss_predicted_autoencoder_second.append(entropy_value_second[idx])
        miss_predicted_autoencoder_first_index.append(idx)
        miss_predicted_autoencoder_first.append(entropy_value_first[idx])
        
        
        
        
        


for idx in range(0, len(cifar_x_test)):
    max_dif=abs(max_probability_first[idx]-max_probability_second[idx])
    entro_dif=abs(entropy_value_first[idx]-entropy_value_second[idx])
          
        
    
    if(entropy_value_first[idx]>=entropy_value_second[idx] and max_probability_first[idx]<=max_probability_second[idx]):
        model_second_x_data.append(cifar_x_test[idx:idx+1])
        model_second_y_data.append(cifar_y_test[idx:idx+1])
        
    elif(entropy_value_first[idx]>entropy_value_second[idx] and max_probability_first[idx]>max_probability_second[idx]):
        if(max_dif>entro_dif):
            model_first_x_data.append(cifar_x_test[idx:idx+1])
            model_first_y_data.append(cifar_y_test[idx:idx+1])
        else:
            model_second_x_data.append(cifar_x_test[idx:idx+1])
            model_second_y_data.append(cifar_y_test[idx:idx+1])
            
    elif(entropy_value_first[idx]<entropy_value_second[idx] and max_probability_first[idx]<max_probability_second[idx]):
        if(max_dif>entro_dif):
            model_second_x_data.append(cifar_x_test[idx:idx+1])
            model_second_y_data.append(cifar_y_test[idx:idx+1])
            
        else:
            model_first_x_data.append(cifar_x_test[idx:idx+1])
            model_first_y_data.append(cifar_y_test[idx:idx+1])
            
            
    #elif(entropy_value_first[idx]<entropy_value_second[idx] and max_probability_first[idx]>max_probability_second[idx)
            
            
            
        
        
    else:    
        model_first_x_data.append(cifar_x_test[idx:idx+1])
        model_first_y_data.append(cifar_y_test[idx:idx+1])
        #print("here")
        
        
model_first_x_data=np.array(model_first_x_data)
#print(type(model_first_x_data))
#print(model_first_x_data.shape)
model_first_x_data=model_first_x_data.reshape(-1,32,32,3)
print(model_first_x_data.shape)
model_first_y_data=np.array(model_first_y_data)
first_y=K.utils.to_categorical(model_first_y_data, 10)
        
model_second_x_data=np.array(model_second_x_data)
model_second_x_data=model_second_x_data.reshape(-1,32,32,3)
print(model_second_x_data.shape)
model_second_y_data=np.array(model_second_y_data)
second_y=K.utils.to_categorical(model_second_y_data, 10)


batch_size = 128
score1 = model_first.evaluate(model_first_x_data, first_y, batch_size=batch_size, verbose=1)
score2 = model_second.evaluate(model_second_x_data, second_y, batch_size=batch_size, verbose=1)



accuarcy = (((score1[1]*len(model_first_x_data))+(score2[1])*len(model_second_x_data))/len(cifar_x_test))

print("Overall accuracy with both autoencoder is : ",accuarcy )


# In[549]:


dat = numpy.array([miss_predicted_autoencoder_first, miss_predicted_autoencoder_second])

dat = dat.T

numpy.savetxt('missed.txt', dat, delimiter = '  ,   ')


# In[320]:


model_second.predict(model_first_x_data[100:101])


# In[405]:


model_first.predict(cifar_x_test[2003:2004])


# In[406]:



np.max(model_first.predict(cifar_x_test[2003:2004]))


# In[407]:


Yv_p[2003:2004]


# In[408]:


model_second.predict(cifar_x_test[2003:2004])


# In[409]:


np.max(model_second.predict(cifar_x_test[2003:2004]))


# In[ ]:





# In[ ]:





# In[411]:


model_second.evaluate(cifar_x_test[2003:2004],Yv_p[2003:2004])


# In[412]:


model_first.evaluate(cifar_x_test[2003:2004],Yv_p[2003:2004])


# In[ ]:





# In[413]:


entropy_value_first[2003:2004]


# In[414]:


entropy_value_second[2003:2004]


# In[256]:


###300 epoch first dissimilar group 
batch_size = 32
score = encoder_model.evaluate(temp_x_test, temp_y, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[258]:


import dill
with open("models/firstgroup_classifier.bn", "wb") as f:
    dill.dump(encoder_model,f)


# In[ ]:





# In[242]:


###100 epoch first dissimilar group 
batch_size = 32
score = encoder_model.evaluate(temp_x_test, temp_y, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])   


# In[ ]:





# In[199]:


######training with same datagroup (second dissimilar group)
batch_size = 32
score = encoder_model.evaluate(temp_x_test, temp_y, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[205]:


#############training with same datagroup (first dissimilar group)
batch_size = 32
score = encoder_model.evaluate(temp_x_test, temp_y, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[218]:


#############training with same datagroup (Animals group)
batch_size = 32
score = encoder_model.evaluate(temp_x_test, temp_y, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[230]:


#############training with same datagroup (Machines group)
batch_size = 32
score = encoder_model.evaluate(temp_x_test, temp_y, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[185]:


######second similar group based autoencoder and using whole training data
batch_size = 32
score = encoder_model.evaluate(temp_x_test, temp_y, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[88]:


batch_size = 32
score = encoder_model.evaluate(cifar_x_test, Yv_p, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[115]:


batch_size = 32
score = encoder_model.evaluate(cifar_x_test, Yv_p, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[308]:


##################compressed#####################
batch_size = 32
score = encoder_model.evaluate(cifar_x_test, Yv_p, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[24]:


batch_size = 32
score = encoder_model.evaluate(cifar_x_test, Yv_p, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[26]:


encoder_model_alexNet= keras.Sequential()


# In[25]:


import keras


# In[29]:


import tensorflow.keras.backend as K


# In[19]:


import tensorflow.keras as K
from keras.preprocessing.image import ImageDataGenerator


# In[304]:


encoder_model_alexNet= K.Sequential()
# using upsamplign to get more data points and improve the predictions
#encoder_model.add(K.layers.UpSampling2D())
encoder_model_alexNet.add(base_encoded)
encoder_model_alexNet.add(K.layers.Flatten())
encoder_model_alexNet.add(BatchNormalization())
#encoder_model_alexNet.add(MaxPooling2D(pool_size=(2, 2), strides=2))
#model.add(Activation('relu'))
encoder_model_alexNet.add(K.layers.Dropout(0.25))  
#encoder_model_alexNet.add(Convolution2D(32, (3, 3), padding='same', activation='relu'))
#model.add(Activation('relu'))
#encoder_model_alexNet.add(MaxPooling2D(pool_size=(2, 2), strides=2))
encoder_model_alexNet.add(K.layers.Dropout(0.15))
encoder_model_alexNet.add(K.layers.Flatten())
encoder_model_alexNet.add(Dense(1024))
encoder_model_alexNet.add(BatchNormalization())
encoder_model_alexNet.add(Dense(512))
encoder_model_alexNet.add(BatchNormalization())
encoder_model_alexNet.add(Activation('relu'))
encoder_model_alexNet.add(K.layers.Dropout(0.25))
encoder_model_alexNet.add(BatchNormalization())
encoder_model_alexNet.add(Dense(256))
encoder_model_alexNet.add(BatchNormalization())
encoder_model_alexNet.add(Activation('tanh'))
encoder_model_alexNet.add(K.layers.Dropout(0.25))
encoder_model_alexNet.add(Dense(10, activation='softmax'))
# adding callbacks
callback = []
callback += [K.callbacks.LearningRateScheduler(decay, verbose=1)]
#callback += [K.callbacks.ModelCheckpoint('cifar10.h5',
#                                         save_best_only=True,
#                                        mode='min'
#                                         )]
# tensorboard callback
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# callback += [K.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]
# Compiling model with adam optimizer and looking the accuracy
#encoder_model_alexNet.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
opt = SGD(lr=0.001, momentum=0.9)
encoder_model_alexNet.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
# training model with mini batch using shuffle data
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
it_train_alex = datagen.flow(temp_x_train, temp_y_train, batch_size=64)
#fit_generator
rs1=encoder_model_alexNet.fit(temp_x_train,temp_y_train,
validation_data=(temp_x_test, temp_y),
epochs=100, shuffle=True,
callbacks=callback,
verbose=1
)


# In[337]:


batch_size = 32
score = encoder_model_alexNet.evaluate(cifar_x_test, Yv_p, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[22]:


batch_size = 32
score = encoder_model_alexNet.evaluate(cifar_x_test, Yv_p, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[437]:


encoder_model_alexNet1= K.Sequential()
# using upsamplign to get more data points and improve the predictions
#encoder_model.add(K.layers.UpSampling2D())
encoder_model_alexNet1.add(base_encoded)
encoder_model_alexNet1.add(Convolution2D(32, (3, 3), padding='same', activation='relu'))
#encoder_model_alexNet.add(MaxPooling2D(pool_size=(2, 2), strides=2))
encoder_model_alexNet1.add(MaxPooling2D((2, 2)))
encoder_model_alexNet1.add(BatchNormalization())
encoder_model_alexNet1.add(K.layers.Dropout(0.25))
encoder_model_alexNet1.add(Convolution2D(32, (3, 3), padding='same', activation='relu'))
#encoder_model_alexNet.add(MaxPooling2D(pool_size=(2, 2), strides=2))

encoder_model_alexNet1.add(BatchNormalization())
encoder_model_alexNet1.add(K.layers.Dropout(0.25))
encoder_model_alexNet1.add(BatchNormalization())
#encoder_model_alexNet1.add(K.layers.Flatten())
#encoder_model_alexNet1.add(BatchNormalization())

#model.add(Activation('relu'))
#encoder_model_alexNet1.add(K.layers.Dropout(0.25))  
#encoder_model_alexNet.add(Convolution2D(32, (3, 3), padding='same', activation='relu'))
#model.add(Activation('relu'))
#encoder_model_alexNet.add(MaxPooling2D(pool_size=(2, 2), strides=2))
encoder_model_alexNet1.add(K.layers.Dropout(0.15))
encoder_model_alexNet1.add(K.layers.Flatten())
encoder_model_alexNet1.add(Dense(1024))
encoder_model_alexNet1.add(BatchNormalization())
encoder_model_alexNet1.add(Dense(512))
encoder_model_alexNet1.add(BatchNormalization())
encoder_model_alexNet1.add(Activation('relu'))
encoder_model_alexNet1.add(K.layers.Dropout(0.25))
encoder_model_alexNet1.add(BatchNormalization())
encoder_model_alexNet1.add(Dense(256))
encoder_model_alexNet1.add(BatchNormalization())
encoder_model_alexNet1.add(Activation('tanh'))
encoder_model_alexNet1.add(K.layers.Dropout(0.25))
encoder_model_alexNet1.add(Dense(10, activation='softmax'))
# adding callbacks
callback = []
callback += [K.callbacks.LearningRateScheduler(decay, verbose=1)]
#callback += [K.callbacks.ModelCheckpoint('cifar10.h5',
#                                         save_best_only=True,
#                                        mode='min'
#                                         )]
# tensorboard callback
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# callback += [K.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]
# Compiling model with adam optimizer and looking the accuracy
encoder_model_alexNet1.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
#opt = SGD(lr=0.001, momentum=0.9)
#encoder_model_alexNet1.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])
# training model with mini batch using shuffle data
it_train_alex = datagen.flow(temp_x_train, temp_y_train, batch_size=64)
rs1=encoder_model_alexNet1.fit(it_train_alex,
validation_data=(temp_x_test, temp_y),
epochs=30, shuffle=True,
callbacks=callback,
verbose=1
)


# In[307]:


encoder_model_alexNet1= K.Sequential()
# using upsamplign to get more data points and improve the predictions
#encoder_model.add(K.layers.UpSampling2D())
encoder_model_alexNet1.add(base_encoded)
encoder_model_alexNet1.add(K.layers.UpSampling2D())
encoder_model_alexNet1.add(K.layers.UpSampling2D())

encoder_model_alexNet1.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
encoder_model_alexNet1.add(BatchNormalization())
encoder_model_alexNet1.add(MaxPooling2D((2, 2)))
encoder_model_alexNet1.add(K.layers.Dropout(0.2))
encoder_model_alexNet1.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
encoder_model_alexNet1.add(BatchNormalization())
encoder_model_alexNet1.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
encoder_model_alexNet1.add(BatchNormalization())
encoder_model_alexNet1.add(MaxPooling2D((2, 2)))
encoder_model_alexNet1.add(K.layers.Dropout(0.3))
encoder_model_alexNet1.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
encoder_model_alexNet1.add(BatchNormalization())
encoder_model_alexNet1.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
encoder_model_alexNet1.add(BatchNormalization())
encoder_model_alexNet1.add(MaxPooling2D((2, 2)))
encoder_model_alexNet1.add(K.layers.Dropout(0.4))
encoder_model_alexNet1.add(K.layers.Flatten())
encoder_model_alexNet1.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
encoder_model_alexNet1.add(BatchNormalization())
encoder_model_alexNet1.add(K.layers.Dropout(0.5))
encoder_model_alexNet1.add(Dense(10, activation='softmax'))
# adding callbacks
callback = []
callback += [K.callbacks.LearningRateScheduler(decay, verbose=1)]
#callback += [K.callbacks.ModelCheckpoint('cifar10.h5',
#                                         save_best_only=True,
#                                        mode='min'
#                                         )]
# tensorboard callback
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# callback += [K.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]
# Compiling model with adam optimizer and looking the accuracy
encoder_model_alexNet1.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
#opt = SGD(lr=0.001, momentum=0.9)
#encoder_model_alexNet1.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])
# training model with mini batch using shuffle data
it_train_alex = datagen.flow(temp_x_train, temp_y_train, batch_size=64)
rs1=encoder_model_alexNet1.fit(it_train_alex,
validation_data=(temp_x_test, temp_y),
epochs=30, shuffle=True,
callbacks=callback,
verbose=1
)


# In[442]:


batch_size = 32
score = encoder_model_alexNet1.evaluate(cifar_x_test, Yv_p, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[333]:


from tensorflow.keras.optimizers import SGD


# In[308]:


encoder_model_alexNet1.summary()


# In[444]:


encoder_model_alexNet.summary()


# In[ ]:


def AlexnetModel_01(input_shape,num_classes):
    model = Sequential()
    model.add(Convolution2D(64, (4, 4), padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.25))
    model.add(Convolution2D(32, (3, 3), padding='same', activation='relu'))
    #model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.15))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64))
    model.add(Activation('tanh'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model


# In[770]:


def compressed_autoencoder():
   dae_inputs = Input(shape=(32, 32, 3), name='dae_input')
   conv_block1 = conv_block(dae_inputs , 32, 3)
   conv_block2 = conv_block(conv_block1, 64, 3)
   conv_block3 = conv_block(conv_block2, 128, 3)
   conv_block4 = conv_block(conv_block3, 256, 3)
   conv_block5 = conv_block(conv_block4, 256, 3, 1)
   #classifier = Dense(nb_classes, activation='softmax')(encoded)
   dae_outputs = Activation('sigmoid', name='dae_output')(conv_block5)
   #autoencoder = Model(dae_inputs, conv_block5)
   return Model(dae_inputs, dae_outputs ,name='dae2')


# In[774]:


dae2.summary()


# In[772]:


dae2 = compressed_autoencoder()
dae2.compile(loss='mse', optimizer='adam')


# In[773]:


compressed = compressed_autoencoder()
compressed.compile(loss='mse', optimizer='adam')

#checkpoint = ModelCheckpoint('best_model.h5', verbose=1, save_best_only=True, save_weights_only=True)

history = dae2.fit_generator(it_train,
                epochs=10,
                shuffle=True,
                validation_data=(val_hard.reshape(len(val_hard), 32, 32,3), val_easy.reshape(len(val_easy),32, 32,3)))


# In[95]:


import dill
with open("models/autoencoder_model_entro.bn", "wb") as f:
    dill.dump(dae,f)


# In[454]:


def add_noise_and_clip_data(data):
   noise = np.random.normal(loc=0.0, scale=0.1, size=data.shape)
   data = data + noise
   data = np.clip(data, 0., 1.)
   return data


# In[3]:


import dill
with open("models/autoencoder_model_entro_latest1.bn", "rb") as f:
         denoise_auto = dill.load(f)


# In[13]:



######Hasan################################################################
######################################################
######################################################
################################################




decoded_images_2nd = denoise_auto.predict(cifar_x_test.reshape(cifar_x_test.shape[0],32,32,3))
autoencode_test2 = np.array(decoded_images_2nd)
autoencode_test2= autoencode_test2.reshape(-1,32,32,3)
batch_size = 32
score = model.evaluate(autoencode_test2, cifar_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[455]:


train_data_noisy =  add_noise_and_clip_data(cifar_x_train)
train_data_clean =  cifar_x_train

test_data_noisy = add_noise_and_clip_data(cifar_x_test)
test_data_clean=  cifar_x_test


# In[524]:


it_train1 = datagen.flow(train_data_noisy, train_data_clean, batch_size=64)


# In[544]:


##params: 
dae1 = denoising_autoencoder()
dae1.compile(loss='mse', optimizer='adam')

#checkpoint = ModelCheckpoint('best_model.h5', verbose=1, save_best_only=True, save_weights_only=True)

history = dae1.fit(train_data_noisy,
                train_data_clean ,
                epochs=100 ,
                shuffle=True,
                validation_data=(test_data_noisy.reshape(len(test_data_noisy),32,32,3), test_data_clean.reshape(len(test_data_clean),32,32,3)))


# In[545]:


decoded_images_2nd = dae1.predict(autoencode_test.reshape(autoencode_test.shape[0],32,32,3))


# In[546]:


decoded_images_2nd.shape


# In[547]:


autoencode_test2 = np.array(decoded_images_2nd)


# In[548]:


autoencode_test2.shape


# In[549]:


autoencode_test2= autoencode_test2.reshape(-1,32,32,3)


# In[550]:


batch_size = 32
score = model.evaluate(autoencode_test2, cifar_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[567]:


batch_size = 32
score = model.evaluate(autoencode_test, cifar_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[235]:


dae.layers[15]


# In[236]:


for layer in dae.layers:
    print(layer.output_shape)


# In[234]:


ouput=dae.layers[15].output


# In[61]:


###params : hard_entropy_data_test , hard_entropy_level_test
decoded_images = dae.predict(cifar_x_test.reshape(cifar_x_test.shape[0],32,32,3))


# In[62]:


decoded_images.shape


# In[63]:


autoencode_test = np.array(decoded_images)


# In[64]:


autoencode_test.shape


# In[230]:





# In[65]:


autoencode_test= autoencode_test.reshape(-1,32,32,3)


# In[254]:





# In[464]:


###without data augmentation and rotation
batch_size = 32
score = model.evaluate(cifar_x_test, cifar_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[66]:


###augmentation and training with hard data only
batch_size = 32
score = model.evaluate(autoencode_test, cifar_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[ ]:





# In[456]:


### augmentation and same number easy data and hard data
batch_size = 32
score = model.evaluate(autoencode_test, cifar_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[143]:


####augmentation and twice number of hard data 
batch_size = 32
score = model.evaluate(autoencode_test, cifar_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[102]:


####augnemtation and twice the number of hard and easy data(both) epoch 100 batchsize 64
batch_size = 32
score = model.evaluate(autoencode_test, cifar_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[145]:


####augnemtation and twice the number of hard and easy data(thrice) epoch 100 batchsize 256
batch_size = 32
score = model.evaluate(autoencode_test, cifar_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[156]:


####300 epochs with twice hard and real_easy data and using branchynet for easy and hard classification
batch_size = 32
score = model.evaluate(autoencode_test, cifar_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[378]:


len(autoencode_test)


# In[233]:


####100 epochs using entropy based classification and also accuracy check
batch_size = 32
score = model.evaluate(autoencode_test, cifar_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[447]:


##300 epochs and increasing data and using entropy value
batch_size = 32
score = model.evaluate(autoencode_test, cifar_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[437]:



batch_size = 32
score = model.evaluate(autoencode_test, cifar_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[384]:


cifar_x_test.shape


# In[370]:


pred = dae.predict(cifar_x_test[5:6])


# In[371]:


plt.imshow(pred.reshape(32,32,3))


# In[375]:


plt.imshow(easy_entropy_data_test[5:6].reshape(32,32,3))


# In[269]:


plt.imshow(autoencode_test[1000:1001].reshape(32,32,3))


# In[ ]:





# In[ ]:


#########################
#device = lightweight model
#data = converted_data
#accuracy = "" similarity
#flops, macs, parameters
##########################



# In[ ]:





# In[ ]:





# In[170]:


plt.imshow(cifar_x_test[1000].reshape(32,32,3))


# In[171]:


plt.imshow(autoencode_test[1000].reshape(32,32,3))


# In[196]:


cifar_y_test[1000]


# In[ ]:





# In[290]:


plt.imshow(cifar_x_test[900].reshape(32,32,3))


# In[291]:


plt.imshow(autoencode_test[900].reshape(32,32,3))


# In[ ]:





# In[297]:


score = model.evaluate(autoencode_test[1000:1001], cifar_y_test[1000:1001], batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[298]:


score = model.evaluate( cifar_x_test[1000:1001], cifar_y_test[1000:1001], batch_size=32, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[314]:


softmaxdata=model.predict(cifar_x_test[1000:1001])
entropyvalue = np.array([entropy(s) for s in softmaxdata]) 


# In[316]:


print(type(entropyvalue))


# In[323]:


def hashmap():
    hashmap={}
    for idx in range(0,len(cifar_x_test)):
        softmaxdata=model.predict(cifar_x_test[idx:idx+1])
        entropyvalue = np.array([entropy(s) for s in softmaxdata])
        h_list=entropy_value.tolist()
        assign = tuple([h_list])
        print(type(idx))
        hashmap[tuple(assign)]=idx
    return hashmap
    


# In[324]:





# In[303]:


print(h)


# In[ ]:





# In[ ]:





# In[ ]:





# In[302]:


print(entropyvalue)


# In[191]:


from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


# In[199]:


cifar_y_test.shape


# In[169]:


score = model.evaluate(autoencode_test[455:456], cifar_y_test[455:456], batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[168]:


score = model.evaluate( cifar_x_test[455:456], cifar_y_test[455:456], batch_size=32, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[116]:


plt.imshow(cifar_x_test[455].reshape(32,32,3))


# In[117]:


plt.imshow(autoencode_test[455].reshape(32,32,3))


# In[ ]:





# In[ ]:





# In[206]:


y_test[455]


# In[60]:


score = model.evaluate(cifar_x_test, cifar_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[ ]:





# In[32]:


from scipy.stats import entropy


# In[208]:


thresholds = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.6, 0.75, 1., 5., 10.]

# calculate the entropy date
def calculate_entropy_value(model,X_test,Y_test):
    xdata = X_test.reshape(-1, 32, 32, 3)
    ydata = Y_test
    softmax_data = model.predict(xdata)

    entropy_value = np.array([entropy(s) for s in softmax_data]) 
    print(entropy_value)
    print(len(entropy_value))
    idx = np.zeros(entropy_value.shape[0],dtype=bool)
    
    #complex_model = load_or_train_model("LeNetModel")
    
    numexits = []
    confidences = []
    no_confidences = []
    total = len(softmax_data)
    print(total)
    for threshold in thresholds:
        idx[entropy_value < threshold] = True
        numexit = sum(idx)
        numexits.append(total - numexit)
        
        # keep no confidence data.
        xdata_keep = xdata[0:10000][~idx[0:10000]]
        ydata_keep = ydata[0:10000][~idx[0:10000]]
        print("confidence ", len(xdata_keep), len(ydata_keep))
        if len(xdata_keep) > 0:
            result = model.evaluate(xdata_keep, y = ydata_keep, batch_size=384)
            print("test loss, test acc:", result, (1-result[1])*len(xdata_keep))
            confidences.append((1-result[1])*len(xdata_keep))

            #complex_result = complex_model.evaluate(xdata_keep, y = ydata_keep, batch_size=384)
            #print("complext models result", complex_result, (1-complex_result[1])*len(xdata_keep))
        else:
            confidences.append(0)
            
        
        # keep confidence data.
        xdata_keep = xdata[idx]
        ydata_keep = ydata[idx]
        print("less confidence", len(xdata_keep), len(ydata_keep))
        if len(xdata_keep) > 0:
            result = model.evaluate(xdata_keep, y = ydata_keep, batch_size=384)
            print("test loss, test acc:", result, (1-result[1])*len(xdata_keep))
            print("")
            no_confidences.append((1-result[1])*len(xdata_keep))
        else:
            no_confidences.append(0)

        
    print(numexits)   
    print("no confidence error data")
    print(confidences)
    print("confidence error data")
    print(no_confidences)


# In[209]:


calculate_entropy_value(model,cifar_x_train,cifar_y_train)


# In[ ]:





# In[197]:


final =[]
missed_final=[]
missed_y_label=[]
count = 0
for x in range(0,len(cifar_x_test)):
    score = model.evaluate(cifar_x_test[x:x+1], cifar_y_test[x:x+1], batch_size=batch_size, verbose=1)
    if(score[1]==1):
        final.append(cifar_x_test[x:x+1])
        print("here")
        
    else:
        img = dae.predict(cifar_x_test[x:x+1])
        final.append(img)
        missed_final.append(cifar_x_test[x:x+1])
        missed_y_label.append(cifar_y_test[x:x+1])
    count+=1
    print(count)


# In[199]:


final_test_set = final.reshape(10000,32,32,3)


# In[198]:


final = np.array(final)


# In[200]:


score = model.evaluate(final_test_set, cifar_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[ ]:





# In[ ]:





# In[201]:


len(missed_final)


# In[202]:


missed_final = np.array(missed_final)


# In[203]:


final_test_miss_set = missed_final.reshape(len(missed_final),32,32,3)


# In[204]:


missed_y_label=np.array(missed_y_label)
missed_y_label=missed_y_label.reshape(len(missed_y_label),1)
missed_y_label=missed_y_label.flatten()


# In[205]:


len(missed_y_label)


# In[206]:


score = model.evaluate(final_test_miss_set, missed_y_label, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


final_test_miss_set[]


# In[207]:


final_test_miss_set=np.array(final_test_miss_set)
xdata = final_test_miss_set.reshape(-1, 32, 32, 3)
#ydata = cifar_y_test
softmax_data = model.predict(xdata)
entropy_value = np.array([entropy(s) for s in softmax_data]) 


# In[208]:


print(entropy_value)


# In[209]:


len(entropy_value)


# In[211]:


for idx in range(0,len(entropy_value)):
    print(entropy_value[idx])


# In[475]:


cifar_x_test=np.array(cifar_x_test)
xdata = cifar_x_test.reshape(-1, 32, 32, 3)
#ydata = cifar_y_test
softmax_data = model.predict(xdata)
entropy_value = np.array([entropy(s) for s in softmax_data]) 


# In[476]:


count =0
test_set=[]
test_ylabel=[]
autoencode_set=[]
autoencode_label=[]
for idx in range(0,len(entropy_value)):
    print(entropy_value[idx])
    if(entropy_value[idx]<0.01):
        count+=1
        test_set.append(cifar_x_test[idx:idx+1])
        test_ylabel.append(cifar_y_test[idx:idx+1])
    else:
        autoencode_set.append(cifar_x_test[idx:idx+1])
        autoencode_label.append(cifar_y_test[idx:idx+1])
        
        
print("Last")        
print(count)


# In[477]:


test_set= np.array(test_set)
test_set= test_set.reshape(-1,32,32,3)


# In[478]:


test_ylabel= np.array(test_ylabel)
test_ylabel.flatten()


# In[479]:


score = model.evaluate(test_set, test_ylabel, batch_size=batch_size, verbose=1)


# In[480]:


score = model.evaluate(test_set, test_ylabel, batch_size=batch_size, verbose=1)


# In[481]:


len(autoencode_set)


# In[482]:


autoencode_set= np.array(autoencode_set)
autoencode_set= autoencode_set.reshape(-1,32,32,3)


# In[483]:


autoencode_label= np.array(autoencode_label)
autoencode_label.flatten()


# In[484]:


len(autoencode_label)


# In[485]:


len(autoencode_set)


# In[144]:


autoencode_set.shape


# In[487]:


decoded_rest = dae.predict(autoencode_set.reshape(autoencode_set.shape[0],32,32,3))


# In[488]:


decoded_rest.shape


# In[ ]:





# In[491]:


score = model.evaluate(decoded_rest, autoencode_label, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[493]:


final_x_set=[]


# In[494]:


final_x_set.append(test_set)
final_x_set.append(decoded_rest)


# In[495]:


len(final_x_set)


# In[270]:


len(test_set)


# In[271]:


len(decoded_rest)


# In[496]:


final_x_set= np.array(final_x_set)


# In[497]:


final_ll= test_set.tolist() + decoded_rest.tolist()


# In[498]:


type(decoded_rest)


# In[499]:


len(final_ll)


# In[500]:


final_ll= np.array(final_ll)


# In[501]:


final_ll= final_ll.reshape(-1,32,32,3)


# In[502]:


test_set= np.array(test_set)


# In[329]:


test_set.shape


# In[330]:


decoded_rest.shape


# In[ ]:





# In[331]:


len(autoencode_label)


# In[332]:


len(test_ylabel)


# In[307]:


type(autoencode_label)


# In[334]:


type(test_ylabel)


# In[503]:


final_y_label = test_ylabel.tolist()+autoencode_label.tolist()


# In[504]:


len(final_y_label)


# In[505]:


final_y_label = np.array(final_y_label)
final_y_label= final_y_label.flatten()


# In[507]:


score = model.evaluate(final_ll, final_y_label, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[ ]:





# In[105]:





# In[623]:


cifar_x_test=np.array(cifar_x_test)
xdata = cifar_x_test.reshape(-1, 32, 32, 3)
#ydata = cifar_y_test
softmax_data = model.predict(xdata)
entropy_value = np.array([entropy(s) for s in softmax_data])
def calculate_thresholded_entropy(thresholds):
    count =0
    test_set=[]
    test_ylabel=[]
    autoencode_set=[]
    autoencode_label=[]
    for idx in range(0,len(entropy_value)):
        #print(entropy_value[idx])
        if(entropy_value[idx]<thresholds):
            count+=1
            test_set.append(cifar_x_test[idx:idx+1])
            test_ylabel.append(cifar_y_test[idx:idx+1])
        else:
            autoencode_set.append(cifar_x_test[idx:idx+1])
            autoencode_label.append(cifar_y_test[idx:idx+1])
        
        
    print("Last")        
    print("Easy Data : ",count)
    print("Hard Data : ",10000-count)
    
    return test_set,test_ylabel,autoencode_set, autoencode_label
    


# In[633]:


def merge_autoencode(thresholds):
    test_set,test_ylabel,autoencode_set,autoencode_label=calculate_thresholded_entropy(thresholds)
    #print(autoencode_set.shape)
    if(test_set and autoencode_set ):
        test_set= np.array(test_set)
        test_set= test_set.reshape(-1,32,32,3)
        test_ylabel= np.array(test_ylabel)
        test_ylabel.flatten()
        autoencode_set=np.array(autoencode_set)
        decoded_rest = dae.predict(autoencode_set.reshape(-1,32,32,3))
        final_ll= test_set.tolist() + decoded_rest.tolist()
        final_y_label = test_ylabel.tolist()+autoencode_label
        final_ll= np.array(final_ll)
        final_ll= final_ll.reshape(-1,32,32,3)
        final_y_label = np.array(final_y_label)
        final_y_label= final_y_label.flatten()
    elif(test_set and not autoencode_set):
        test_set= np.array(test_set)
        test_set= test_set.reshape(-1,32,32,3)
        test_ylabel= np.array(test_ylabel)
        test_ylabel.flatten()
        final_ll= test_set.tolist()
        final_y_label = test_ylabel.tolist()
        final_ll= np.array(final_ll)
        final_ll= final_ll.reshape(-1,32,32,3)
        final_y_label = np.array(final_y_label)
        final_y_label= final_y_label.flatten()
    elif(autoencode_set and not test_set):
        autoencode_set=np.array(autoencode_set)
        decoded_rest = dae.predict(autoencode_set.reshape(-1,32,32,3))
        final_ll= autoencode_set.tolist()
        final_y_label = autoencode_label
        final_ll= np.array(final_ll)
        final_ll= final_ll.reshape(-1,32,32,3)
        final_y_label = np.array(final_y_label)
        final_y_label= final_y_label.flatten()
    
        
    return final_ll,final_y_label,test_set,test_ylabel
    
    
    


# In[605]:


best_accuarcy = 0.0
best_thresholds=0


# In[636]:


def thresholded_accuracy():
    
    thresholds_array = np.linspace(0.001, 1, num=100)
    best_accuarcy = 0.0
    best_thresholds = 0
    for thresholds in (thresholds_array):
        final_ll,final_y_label,test_set,test_ylabel = merge_autoencode(thresholds)
    
        score = model.evaluate(final_ll, final_y_label, batch_size=batch_size, verbose=1)
        print("For threshold :",thresholds)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        if(score[1]>best_accuarcy):
            best_accuarcy=score[1]
            best_thresholds=thresholds
        
        print()
    
    return best_accuarcy,best_thresholds
    
    


# In[637]:


best_acc,best_th=thresholded_accuracy()


# In[ ]:


original =82%
original+ model : 84.82%
    
we have lightweight model, 


# In[638]:


best_acc,best_th


# In[597]:




final_ll,final_y_label,test_set,test_ylabel=merge_autoencode(.01)


# In[598]:


score = model.evaluate(test_set, test_ylabel, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[599]:


score = model.evaluate(final_ll, final_y_label, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[ ]:





# In[400]:


softmax_data_temp = model.predict(final_ll[1000:1001])
entropy_value_temp = np.array([entropy(s) for s in softmax_data_temp])


# In[402]:


print(entropy_value_temp)


# In[113]:


final_entropy =[]
count = 0
for x in range(0,len(cifar_x_test)):
    #score = model.evaluate(cifar_x_test[x:x+1], cifar_y_test[x:x+1], batch_size=batch_size, verbose=1)
    softmax_data_temp = model.predict(cifar_x_test[x:x+1])
    entropy_value_temp = np.array([entropy(s) for s in softmax_data_temp])
    decoded_auto_entropy = dae.predict(cifar_x_test[x:x+1])
    softmax_data_temp_decoded = model.predict(decoded_auto_entropy)
    entropy_value_temp_decoded = np.array([entropy(s) for s in softmax_data_temp_decoded])
    if(entropy_value_temp>entropy_value_temp_decoded):
        final_entropy.append(decoded_auto_entropy)
        print("here")
        
    else:
        final_entropy.append(cifar_x_test[x:x+1])
    count+=1
    print(count)


# In[ ]:





# In[114]:


final_entropy=np.array(final_entropy)


# In[116]:


print(len(final_entropy))
final_entropy= final_entropy.reshape(-1,32,32,3)


# In[117]:


final_entropy.shape


# In[118]:


score = model.evaluate(final_entropy, cifar_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[119]:


plt.imshow(final_entropy[455:456].reshape(32,32,3))


# In[120]:


score = model.evaluate(final_entropy[455:456], cifar_y_test[455:456], batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[586]:


def plane_Bucket_cluster(img1,img2):
        test_image1 = img1.reshape(32,32,3)
        test_image1 = test_image1.flatten()
        test_image1 = test_image1/255
        test_image2 = img2.reshape(32,32,3)
        test_image2 = test_image2.flatten()
        test_image2 = test_image2/255
        similarity = -1 * (spatial.distance.cosine(test_image1, test_image2) - 1)
        #x= x+1
        return similarity


# In[124]:


plane_Bucket_cluster(cifar_x_test[1205:1206],final_entropy[1203:1204])


# In[123]:


plt.imshow(final_entropy[1203:1204].reshape(32,32,3))


# In[ ]:


plt.imshow(cifar_x_test[1205:1206].reshape(32,32,3))


# In[ ]:





# In[588]:


from scipy import spatial


# In[ ]:





# In[589]:


final_entropy_cosine =[]
count = 0
count_entropy=0
for x in range(0,len(cifar_x_test)):
    #score = model.evaluate(cifar_x_test[x:x+1], cifar_y_test[x:x+1], batch_size=batch_size, verbose=1)
    softmax_data_temp = model.predict(cifar_x_test[x:x+1])
    entropy_value_temp = np.array([entropy(s) for s in softmax_data_temp])
    decoded_auto_entropy = dae.predict(cifar_x_test[x:x+1])
    softmax_data_temp_decoded = model.predict(decoded_auto_entropy)
    entropy_value_temp_decoded = np.array([entropy(s) for s in softmax_data_temp_decoded])
    similarity= plane_Bucket_cluster(cifar_x_test[x:x+1],decoded_auto_entropy)
    print(similarity)
    if(entropy_value_temp>entropy_value_temp_decoded and similarity>0.94):
        final_entropy_cosine.append(decoded_auto_entropy)
        print("here")
        count_entropy+=1
        
    else:
        final_entropy_cosine.append(cifar_x_test[x:x+1])
    count+=1
    print(count)


# In[590]:


print(count_entropy)


# In[591]:


final_entropy_cosine=np.array(final_entropy_cosine)


# In[592]:


print(len(final_entropy_cosine))
final_entropy_cosine= final_entropy_cosine.reshape(-1,32,32,3)


# In[593]:


final_entropy_cosine.shape


# In[594]:


score = model.evaluate(final_entropy_cosine, cifar_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[583]:


score = model.evaluate(cifar_x_test, cifar_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[429]:


plt.imshow(final_entropy[455:456].reshape(32,32,3))


# In[430]:


plt.imshow(cifar_x_test[455:456].reshape(32,32,3))


# In[471]:


len(train_exits)
train_easy=[]
train_hard=[]


# In[475]:


train_easy=[]
train_hard=[]
train_easy_label=[]
train_hard_label=[]
for idx in range(0,len(train_exits)):
    if(train_exits[idx]==0):
        train_easy.append(cifar_x_train[idx])
        train_easy_label.append(cifar_y_train[idx])
    else:
        train_hard.append(cifar_x_train[idx])
        train_hard_label.append(cifar_y_train[idx])
        


# In[483]:


train_easy= np.array(train_easy)
train_easy= train_easy.reshape(-1,32,32,3)
train_hard= np.array(train_hard)
train_hard= train_hard.reshape(-1,32,32,3)


# In[484]:


train_easy_label=np.array(train_easy_label)
train_easy_label=train_easy_label.flatten()
train_hard_label=np.array(train_hard_label)
train_hard_label=train_hard_label.flatten()


# In[485]:


score = model.evaluate(train_easy, train_easy_label, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[487]:


score = model.evaluate(train_hard, train_hard_label, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[479]:


len(train_easy)


# In[480]:


len(train_easy_label)


# In[481]:


print(train_easy_label)


# In[488]:


train_easy.shape


# In[ ]:





# In[ ]:





# In[17]:


from keras.models import Sequential
import numpy as np
import os
from keras.datasets import cifar10, cifar100
from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.layers import Dropout
from keras.layers import Flatten, Activation
from keras.constraints import maxnorm
from tensorflow.keras.optimizers import SGD, Adam
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras import backend as K

from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
get_ipython().run_line_magic('matplotlib', 'inline')

import skimage
from skimage.util import img_as_ubyte
from scipy.stats import entropy


# In[572]:


def AlexnetModel(input_shape,num_classes):
    model = Sequential()
    model.add(Convolution2D(64, (4, 4), padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    #model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(64, (2, 2), padding='same', activation='relu'))
    #model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    
    model.add(Convolution2D(32, (3, 3), padding='same', activation='relu'))
    #model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(32, (3, 3), padding='same', activation='relu'))
    #model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.15))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64))
    model.add(Activation('tanh'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model


# In[ ]:





# In[560]:


input_shape=(32,32,3)
num_classes =10


# In[561]:


def AlexnetModel_01(input_shape,num_classes):
    model = Sequential()
    model.add(Convolution2D(64, (4, 4), padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    #model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    
    model.add(Convolution2D(32, (3, 3), padding='same', activation='relu'))
    #model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.15))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64))
    model.add(Activation('tanh'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model


# In[562]:


lightweight_model = AlexnetModel_01(input_shape,num_classes)


# In[564]:


lightweight_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[571]:


lightweight_model.summary()


# In[565]:


rs = lightweight_model.fit(
  cifar_x_train, cifar_y_train, validation_data=(cifar_x_test, cifar_y_test), epochs=50)


# In[573]:


alex_net_model = AlexnetModel(input_shape,num_classes)


# In[578]:


alex_net_model.summary()


# In[574]:


alex_net_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[575]:


alex = alex_net_model.fit(
  cifar_x_train, cifar_y_train, validation_data=(cifar_x_test, cifar_y_test), epochs=50)


# In[576]:


score = alex_net_model.evaluate(cifar_x_test, cifar_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[577]:


score = alex_net_model.evaluate(autoencode_test, cifar_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[568]:


score = lightweight_model.evaluate(autoencode_test, cifar_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[570]:


score = lightweight_model.evaluate(cifar_x_test, cifar_y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[ ]:





# In[ ]:





# In[446]:


import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
from tensorflow import keras
import keras.layers as layers
alexnet_model1 = keras.Sequential()
alexnet_model1.add(layers.Conv2D(filters=96, kernel_size=(11, 11), 
                        strides=(4, 4), activation="relu", 
                        input_shape=(32, 32, 3)))
alexnet_model1.add(layers.BatchNormalization())
alexnet_model1.add(layers.MaxPool2D(pool_size=(3, 3), strides= (2, 2)))
alexnet_model1.add(layers.Conv2D(filters=256, kernel_size=(5, 5), 
                        strides=(1, 1), activation="relu", 
                        padding="same"))
alexnet_model1.add(layers.BatchNormalization())
alexnet_model1.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
alexnet_model1.add(layers.Conv2D(filters=384, kernel_size=(3, 3), 
                        strides=(1, 1), activation="relu", 
                        padding="same"))
alexnet_model1.add(layers.BatchNormalization())
alexnet_model1.add(layers.Conv2D(filters=384, kernel_size=(3, 3), 
                        strides=(1, 1), activation="relu", 
                        padding="same"))
alexnet_model1.add(layers.BatchNormalization())
alexnet_model1.add(layers.Conv2D(filters=256, kernel_size=(3, 3), 
                        strides=(1, 1), activation="relu", 
                        padding="same"))
alexnet_model1.add(layers.BatchNormalization())
alexnet_model1.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
alexnet_model1.add(layers.Flatten())
alexnet_model1.add(layers.Dense(4096, activation="relu"))
alexnet_model1.add(layers.Dropout(0.5))
alexnet_model1.add(layers.Dense(10, activation="softmax"))
alexnet_model1.compile(loss='sparse_categorical_crossentropy', 
              optimizer=tf.optimizers.SGD(lr=0.001), 
              metrics=['accuracy'])
alexnet_model1.summary()


# In[ ]:





# In[432]:


def vgg_net():
    model = K.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(K.layers.Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(K.layers.Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(K.layers.Dropout(0.4))
    model.add(K.layers.Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(K.layers.Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    # compile model
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    return model 


# In[433]:


vgg_net= vgg_net()


# In[434]:


vgg_net.build((32,32,3)) 


# In[431]:


vgg_net.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[435]:


vgg_net.summary()


# In[ ]:





# In[33]:



import tensorflow.keras as K


# In[34]:


model = K.Sequential()


# In[ ]:





# In[122]:


check = K.Sequential()
check.add(Conv2D(10, (5,5), input_shape=(64, 6, 6)))
check.add(Conv2D(10,(1,1)))
check.add(K.layers.UpSampling2D())
check.add(Conv2D(10,(2,2)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[123]:


check.summary()


# In[415]:


max_probability_second =[]
for idx in range(0,len(cifar_x_test)):
    max_probability_second.append(np.max(model_second.predict(cifar_x_test[idx:idx+1])))
    


# In[416]:


max_probability_first =[]
for idx in range(0,len(cifar_x_test)):
    max_probability_first.append(np.max(model_first.predict(cifar_x_test[idx:idx+1])))


# In[514]:


type(max_probability_first)


# In[417]:


dat = numpy.array([max_probability_first, max_probability_second])

dat = dat.T

numpy.savetxt('data.txt', dat, delimiter = '  ,   ')


# In[424]:


max_probability_second[6:7]


# In[425]:


type(entropy_value_first[idx])


# In[ ]:


>=entropy_value_second[idx]


# In[426]:


entropy_value_first_list= entropy_value_first.tolist()


# In[427]:


entropy_value_second_list= entropy_value_second.tolist()


# In[428]:


dat = numpy.array([max_probability_first, max_probability_second,entropy_value_first_list,entropy_value_second_list])

dat = dat.T

numpy.savetxt('data.txt', dat, delimiter = '  ,   ')


# In[443]:


similar_animals=[2,3,4,5,6,7]
similar_machines=[0,1,8,9]
animal_x_train=[]
animals_y_train=[]
machine_x_train=[]
machine_y_train=[]

for idx in range(0,len(cifar_x_train)):
    if(cifar_y_train[idx] in similar_animals):
        animal_x_train.append(cifar_x_train[idx:idx+1])
        animals_y_train.append(0)
        
    else:
        machine_x_train.append(cifar_x_train[idx:idx+1])
        machine_y_train.append(0)
        
        
        
        


# In[451]:


similar_animals=[2,3,4,5,6,7]
similar_machines=[0,1,8,9]
animal_x_train=[]
animals_y_train=[]
machine_x_train=[]
machine_y_train=[]
y_train_heirchial=[]

for idx in range(0,len(cifar_x_train)):
    if(cifar_y_train[idx] in similar_animals):
        #animal_x_train.append(cifar_x_train[idx:idx+1])
        y_train_heirchial.append(0)
        
    else:
        #machine_x_train.append(cifar_x_train[idx:idx+1])
        y_train_heirchial.append(1)


# In[453]:


similar_animals=[2,3,4,5,6,7]
similar_machines=[0,1,8,9]
animal_x_train=[]
animals_y_train=[]
machine_x_train=[]
machine_y_train=[]
y_test_heirchial=[]

for idx in range(0,len(cifar_x_test)):
    if(cifar_y_test[idx] in similar_animals):
        #animal_x_train.append(cifar_x_train[idx:idx+1])
        y_test_heirchial.append(0)
        
    else:
        #machine_x_train.append(cifar_x_train[idx:idx+1])
        y_test_heirchial.append(1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[454]:


len(y_test_heirchial)


# In[ ]:


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


def denoising_autoencoder():
   dae_inputs = Input(shape=(32, 32, 3), name='dae_input')
   conv_block1 = conv_block(dae_inputs, 32, 3)
   conv_block2 = conv_block(conv_block1, 64, 3)
   conv_block3 = conv_block(conv_block2, 128, 3)
   conv_block4 = conv_block(conv_block3, 256, 3)
   conv_block5 = conv_block(conv_block4, 256, 3, 1)

   deconv_block1 = deconv_block(conv_block5, 256, 3)
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


# In[481]:


dae.summary()


# In[452]:


len(y_train_heirchial)


# In[11]:






######New classifier
similar_first =[0,1,2,3,7]
similar_2nd= [4,5,6,8,9]
similar_animals=[2,3,4,5,6,7]
similar_machines=[0,1,8,9]
animal_x_train=[]
animals_y_train=[]
machine_x_train=[]
machine_y_train=[]
y_test_heirchial_group=[]

for idx in range(0,len(cifar_x_test)):
    if(cifar_y_test[idx] in similar_first):
        #animal_x_train.append(cifar_x_train[idx:idx+1])
        y_test_heirchial_group.append(0)
        
    else:
        #machine_x_train.append(cifar_x_train[idx:idx+1])
        y_test_heirchial_group.append(1)


# In[12]:


similar_first =[0,1,2,3,7]
similar_2nd= [4,5,6,8,9]
similar_animals=[2,3,4,5,6,7]
similar_machines=[0,1,8,9]
animal_x_train=[]
animals_y_train=[]
machine_x_train=[]
machine_y_train=[]
y_train_heirchial_group=[]

for idx in range(0,len(cifar_x_train)):
    if(cifar_y_train[idx] in similar_first):
        #animal_x_train.append(cifar_x_train[idx:idx+1])
        y_train_heirchial_group.append(0)
        
    else:
        #machine_x_train.append(cifar_x_train[idx:idx+1])
        y_train_heirchial_group.append(1)


# In[13]:


y_test_heirchial_group=np.array(y_test_heirchial_group)
y_test_heirchial_group=y_test_heirchial_group.reshape(10000,)
y_train_heirchial_group=np.array(y_train_heirchial_group)
y_train_heirchial_group=y_train_heirchial_group.reshape(50000,)


# In[14]:


import tensorflow.keras as K


# In[15]:


Y_train=  K.utils.to_categorical(y_train_heirchial_group, 2)
Y_test=  K.utils.to_categorical(y_test_heirchial_group, 2)


# In[624]:


y_test_heirchial_group.shape


# In[627]:


y_train_heirchial_group.shape


# In[579]:


Y_test.shape


# In[ ]:


Classify_model.fit(cifar_x_train, Y_train,
batch_size=128,
validation_data=(cifar_x_test, Y_test),
epochs=30,
verbose=1
)


# In[16]:


cifar_x_train.shape


# In[17]:


auto_x_train=cifar_x_train.reshape(50000,-1)
auto_x_test= cifar_x_test.reshape(10000,-1)


# In[18]:


auto_x_train.shape
auto_x_test.shape


# In[19]:


from autokeras import StructuredDataClassifier


# In[ ]:


#batch_size=128,num_features=100 ,
from autokeras import StructuredDataClassifier
search = StructuredDataClassifier(max_trials=60)
search.fit(x=auto_x_train, y=y_train_heirchial_group, verbose=1)


# In[2]:


loss, acc = search.evaluate(auto_x_test, y_test_heirchial_group, verbose=1)


# In[1]:


print("something")


# In[647]:


Classify_model.predict(cifar_x_test[1:2])


# In[665]:


Classify_model.fit(cifar_x_train, Y_train,
batch_size=128,
validation_data=(cifar_x_test, Y_test),
epochs=50,
verbose=1
)


# In[ ]:





# In[646]:


import dill

with open("models/Autoencoder_Detect_model.bn", "wb") as f:
    dill.dump(Classify_model, f)


# In[ ]:


#################New Accuacry#####################

Data=np.array(cifar_x_test)
xdata = Data.reshape(-1, 32, 32, 3)
#ydata = cifar_y_test
softmax_data1 = model_first.predict(xdata)
#print(softmax_data)
entropy_value_first = np.array([entropy(s) for s in softmax_data1])
softmax_data2 = model_second.predict(xdata)
#print(softmax_data)
entropy_value_second = np.array([entropy(s) for s in softmax_data2])

model_first_x_data=[]
model_first_y_data=[]
model_second_x_data=[]
model_second_y_data=[]
miss_predicted_autoencoder_first=[]
miss_predicted_autoencoder_second=[]
miss_predicted_autoencoder_first_index=[]
miss_predicted_autoencoder_second_index=[]




#entropy_value_first[idx]>entropy_value_second[idx] and 
#max_probability_first[idx]>max_probability_second[idx]

for idx in range(0, len(cifar_x_test)):
    score1 = alex_net_model.evaluate(autoencode_test, cifar_y_test, batch_size=batch_size, verbose=1)
    
    
    
    

        
        
model_first_x_data=np.array(model_first_x_data)
#print(type(model_first_x_data))
#print(model_first_x_data.shape)
model_first_x_data=model_first_x_data.reshape(-1,32,32,3)
print(model_first_x_data.shape)
model_first_y_data=np.array(model_first_y_data)
first_y=K.utils.to_categorical(model_first_y_data, 10)
        
model_second_x_data=np.array(model_second_x_data)
model_second_x_data=model_second_x_data.reshape(-1,32,32,3)
print(model_second_x_data.shape)
model_second_y_data=np.array(model_second_y_data)
second_y=K.utils.to_categorical(model_second_y_data, 10)


batch_size = 128
score1 = model_first.evaluate(model_first_x_data, first_y, batch_size=batch_size, verbose=1)
score2 = model_second.evaluate(model_second_x_data, second_y, batch_size=batch_size, verbose=1)



accuarcy = (((score1[1]*len(model_first_x_data))+(score2[1])*len(model_second_x_data))/len(cifar_x_test))

print("Overall accuracy with both autoencoder is : ",accuarcy )


# In[ ]:





# In[ ]:





# In[455]:


import tensorflow.keras as K
import datetime
def preprocess_data(X, Y):
    """ This method has the preprocess to train a model """
    # applying astype to change float64 to float32 for version 1.12
    # X = X.astype('float32')
    #using preprocess VGG16 method by default to scale images and their values
    X_p = K.applications.vgg16.preprocess_input(X)
    # changind labels to one-hot representation
    Y_p = K.utils.to_categorical(Y, 10)
    return (X_p, Y_p)

def decay(epoch):
    """ This method create the alpha"""
    # returning a very small constant learning rate
    return 0.001 / (1 + 1 * 30)

# loading data and using preprocess for training and validation dataset
Yt=  K.utils.to_categorical(y_train_heirchial, 10)
Y=  K.utils.to_categorical(y_test_heirchial, 10)

#X=X/255.0

###original
#X_p, Y_p = preprocess_data(Xt, Yt)
#Xv_p, Yv_p = preprocess_data(X, Y)



#X_p=Xt.reshape(-1,32,32,3)
#Xv_p=Xv_p.reshape(-1,32,32,3)
#X_p=X_p/255.0
#Xv_p=Xv_p/255.0
# Getting the model without the last layers, trained with imagenet and with average pooling


# In[ ]:





# In[601]:


base_model = K.applications.vgg16.VGG16(include_top=False,
weights='imagenet',
pooling='avg',
input_shape=(32,32,3)
)
# create the new model applying the base_model (VGG16)
model= K.Sequential()
# using upsamplign to get more data points and improve the predictions
#model.add(K.layers.UpSampling2D())
model.add(base_model)
model.add(K.layers.Flatten())
model.add(K.layers.Dense(512, activation=('relu')))
model.add(K.layers.Dropout(0.2))
model.add(K.layers.Dense(256, activation=('relu')))
model.add(K.layers.Dropout(0.2))
model.add(K.layers.Dense(2, activation=('softmax')))
# adding callbacks
callback = []
callback += [K.callbacks.LearningRateScheduler(decay, verbose=1)]
#callback += [K.callbacks.ModelCheckpoint('cifar10.h5',
#                                         save_best_only=True,
#                                        mode='min'
#                                         )]
# tensorboard callback
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# callback += [K.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]
# Compiling model with adam optimizer and looking the accuracy
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
# training model with mini batch using shuffle data
rs=model.fit(cifar_x_train, y_train_heirchial,
batch_size=128,
validation_data=(cifar_x_test, y_test_heirchial),
epochs=30, shuffle=True,
callbacks=callback,
verbose=1
)


# In[609]:


def vgg_net():
    model = K.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(K.layers.Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(K.layers.Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(K.layers.Dropout(0.4))
    model.add(K.layers.Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(K.layers.Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    # compile model
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    return model


# In[610]:


Classify_model= vgg_net()


# In[497]:


y_train_heirchial.shape


# In[484]:


Classify_model.summary()


# In[ ]:


Yt=  K.utils.to_categorical(y_train_heirchial, 10)
Y=  K.utils.to_categorical(y_test_heirchial, 10)


# In[466]:


y_train_heirchial=np.array(y_train_heirchial).reshape(50000,1)
y_test_heirchial=np.array(y_test_heirchial).reshape(10000,1)


# In[504]:


y_test_heirchial.shape


# In[508]:





# In[500]:


cifar_y_test.shape


# In[511]:


Classify_model.fit(cifar_x_train, y_train_heirchial,
batch_size=128,
validation_data=(cifar_x_test, y_test_heirchial),
epochs=30, shuffle=True,
callbacks=callback,
verbose=1
)


# In[502]:


y_test_heirchial[1000:1001]


# In[471]:


score = Classify_model.evaluate(cifar_x_test, y_test_heirchial, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[634]:


K = len(set(y_train))
 
# calculate total number of classes
# for output layer
print("number of classes:", K)
 
# Build the model using the functional API
# input layer
i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
 
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
 
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
 
x = Flatten()(x)
x = Dropout(0.2)(x)
 
# Hidden layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
 
# last hidden layer i.e.. output layer
x = Dense(2, activation='softmax')(x)
 
alex_Model = Model(i, x)
 
# model description
alex_Model.summary()


# In[474]:


alex_Model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
alex_Model.fit(cifar_x_train, y_train_heirchial,
batch_size=128,
validation_data=(cifar_x_test, y_test_heirchial),
epochs=30, shuffle=True,
callbacks=callback,
verbose=1
)


# In[491]:


alex_Model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
alex_Model.fit(cifar_x_train, y_train_heirchial,
batch_size=128,
validation_data=(cifar_x_test, y_test_heirchial),
epochs=30, shuffle=True,
callbacks=callback,
verbose=1
)


# In[ ]:


Classify_model.summary()


# In[476]:


import tensorflow.keras as K


# In[478]:


temp_y = K.utils.to_categorical(cifar_y_train, 10)


# In[479]:


Data=np.array(cifar_x_test)
xdata = Data.reshape(-1, 32, 32, 3)
#ydata = cifar_y_test
softmax_data1 = model_first.predict(xdata)
#print(softmax_data)
entropy_value_first = np.array([entropy(s) for s in softmax_data1])
softmax_data2 = model_second.predict(xdata)
#print(softmax_data)
entropy_value_second = np.array([entropy(s) for s in softmax_data2])

model_first_x_data=[]
model_first_y_data=[]
model_second_x_data=[]
model_second_y_data=[]

miss_predicted_entropy_first=[]
miss_predicted_entropy_second=[]
miss_predicted_out=[]


for idx in range(0, len(cifar_x_test)):
    if(entropy_value_first[idx]>=entropy_value_second[idx]):
        score1 = model_first.evaluate(cifar_x_test[idx:idx+1], temp_y[idx:idx+1], batch_size=batch_size, verbose=1)
        score2 = model_second.evaluate(cifar_x_test[idx:idx+1], temp_y[idx:idx+1], batch_size=batch_size, verbose=1)
        if(score1[1]>score2[1]):
           miss_predicted_entropy_first.append(entropy_value_first[idx:idx+1]) 
           miss_predicted_entropy_second.append(entropy_value_second[idx:idx+1])
           miss_predicted_out.append(1)
    else:
        score1 = model_first.evaluate(cifar_x_test[idx:idx+1], temp_y[idx:idx+1], batch_size=batch_size, verbose=1)
        score2 = model_second.evaluate(cifar_x_test[idx:idx+1], temp_y[idx:idx+1], batch_size=batch_size, verbose=1)
        if(score2[1]>score1[1]):
            miss_predicted_entropy_first.append(entropy_value_first[idx:idx+1]) 
            miss_predicted_entropy_second.append(entropy_value_second[idx:idx+1])
            miss_predicted_out.append(2)
            


# In[ ]:


dat = numpy.array([miss_predicted_entropy_first, miss_predicted_entropy_second,miss_predicted_out])

dat = dat.T

numpy.savetxt('miss_predicted.txt', dat, delimiter = '  ,   ')


# In[ ]:


batch_size = 128
score1 = model_first.evaluate(model_first_x_data, first_y, batch_size=batch_size, verbose=1)
score2 = model_second.evaluate(model_second_x_data, second_y, batch_size=batch_size, verbose=1)


# In[ ]:





# In[ ]:


model_first_x_data=np.array(model_first_x_data)
#print(type(model_first_x_data))
#print(model_first_x_data.shape)
model_first_x_data=model_first_x_data.reshape(-1,32,32,3)
print(model_first_x_data.shape)
model_first_y_data=np.array(model_first_y_data)
first_y=K.utils.to_categorical(model_first_y_data, 10)
        
model_second_x_data=np.array(model_second_x_data)
model_second_x_data=model_second_x_data.reshape(-1,32,32,3)
print(model_second_x_data.shape)
model_second_y_data=np.array(model_second_y_data)
second_y=K.utils.to_categorical(model_second_y_data, 10)


batch_size = 128
score1 = model_first.evaluate(model_first_x_data, first_y, batch_size=batch_size, verbose=1)
score2 = model_second.evaluate(model_second_x_data, second_y, batch_size=batch_size, verbose=1)



accuarcy = (((score1[1]*len(model_first_x_data))+(score2[1])*len(model_second_x_data))/len(cifar_x_test))

print("Overall accuracy with both autoencoder is : ",accuarcy )


# In[669]:


from autokeras import StructuredDataRegressor


# In[668]:


get_ipython().system('pip3 install autokeras')


# In[ ]:




