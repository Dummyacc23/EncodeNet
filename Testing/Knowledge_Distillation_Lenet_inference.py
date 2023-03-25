#!/usr/bin/env python
# coding: utf-8

# In[39]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
# metrics 
from keras.metrics import categorical_crossentropy
# optimization method
from tensorflow.keras.optimizers import SGD
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data
x_train = x_train.astype("float32") / 255.0
x_train = np.reshape(x_train, (-1, 28, 28, 1))

x_test = x_test.astype("float32") / 255.0
x_test = np.reshape(x_test, (-1, 28, 28, 1))
class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)

            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            distillation_loss = (
                self.distillation_loss_fn(
                    tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                    tf.nn.softmax(student_predictions / self.temperature, axis=1),
                )
                * self.temperature**2
            )

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results
    
    
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


LeNet_Teacher= LeNet()

LeNet_Teacher.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
def LeNet_Pruned():
        model = Sequential()  
        model.add(Conv2D(filters = 3, kernel_size = (5,5), padding = 'same', strides = 1,  activation = 'relu', input_shape = (28,28,1)))
        # Max-pooing layer with pooling window size is 2x2
        model.add(MaxPooling2D(pool_size = (2,2)))
        # Convolutional layer 
        model.add(Conv2D(filters = 5, kernel_size = (5,5), padding = 'same', strides = 1,activation = 'relu'))
        # Max-pooling layer 
        model.add(MaxPooling2D(pool_size = (2,2)))
        # Flatten layer 
        model.add(Conv2D(filters = 7, kernel_size = (5,5), padding = 'same', strides = 1,activation = 'relu'))
        model.add(Flatten())

        # The first fully connected layer 
        model.add(Dense(32, activation = 'relu'))

        # The output layer  
        model.add(Dense(10, activation = 'softmax'))

        # compile the model with a loss function, a metric and an optimizer function
        # In this case, the loss function is categorical crossentropy, 
        # we use Stochastic Gradient Descent (SGD) method with learning rate lr = 0.01 
        # metric: accuracy 

        

        return model
    
    
Student_LeNet= LeNet_Pruned()


distiller = Distiller(student=Student_LeNet, teacher=LeNet_Teacher)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
)
distiller.load_weights('knowledge_distillation_lenet')

import time
start = time.time()
score = distiller.evaluate(x_test, y_test, verbose=0)
end = time.time()
print("Time Taken by Knowledge Distillation For lenet",end-start)
print('Test loss:', score[1])
print('Test accuracy:', score[0])


# In[38]:





# In[7]:





# In[8]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




