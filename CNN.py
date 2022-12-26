#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries 

import numpy as np

from tensorflow import keras
import keras

import matplotlib.pyplot as plt

from keras import backend as K
from keras import applications

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Input, add, Add
from keras.layers import Activation, concatenate, AveragePooling2D, BatchNormalization, ZeroPadding2D
from keras.layers import ReLU, DepthwiseConv2D, GlobalAveragePooling2D

from keras.models import Sequential
from keras.models import Model

from keras.regularizers import l2

from keras.optimizers import SGD

from keras import layers,models,losses

import torch
from torch import nn


# # Topic 1: LeNet

# ![index.png](attachment:index.png)

# In[2]:


#Done - 1

model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5), padding = 'same', activation='sigmoid', input_shape = (28,28,1)))
model.add(AveragePooling2D(pool_size=(2,2),strides=2))
#model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(16, kernel_size = (5,5),activation='sigmoid'))
# model.add(MaxPooling2D(pool_size = (2,2)))
model.add(AveragePooling2D(pool_size=(2,2),strides=2))


model.add(Flatten())
model.add(Dense(120,activation = 'sigmoid'))
model.add(Dense(84,activation = 'sigmoid'))
# model.add(Dropout(0.3))

model.add(Dense(10))

model.summary()


# ![LENET_MNIST_model_plot.png](attachment:LENET_MNIST_model_plot.png)

# ### Image : LeNet

# ![LeNet5_800px_web.jpg](attachment:LeNet5_800px_web.jpg)

# In[3]:


#Done - 2

model = Sequential()

model.add(Conv2D(6, kernel_size=(5, 5), strides=1, activation='relu', input_shape = (32,32,1)))
model.add(AveragePooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(16, kernel_size=(5, 5), strides=1, activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2),strides=2))

model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))

model.add(Dense(10, activation="softmax"))

model.summary()


# # Topic 2: CNN on CIFAR10 Data

# In[4]:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()


# ### CNN Mnist Keras Model

# ![3.CNN_MNIST_KERAS_model_plot.png](attachment:3.CNN_MNIST_KERAS_model_plot.png)

# In[5]:


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)
batch_size = 128


# In[6]:


model = keras.Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation="softmax"))

model.summary()


# # Topic 3: AlexNet

# ![AlexNet.png](attachment:AlexNet.png)

# In[7]:


#Done - 3 

#Model Building
model = Sequential()
model.add(Conv2D(filters = 96, kernel_size = 11,strides = 4,padding = 'valid',activation = 'relu',input_shape = (224,224,3)))
model.add(MaxPooling2D(pool_size = 3, strides = 2,padding ='valid'))
# model.add(BatchNormalization())

model.add(Conv2D(filters = 256, kernel_size = 5,strides = 1,padding = 'same',activation = 'relu'))
model.add(MaxPooling2D(pool_size = 3, strides = 2))
# model.add(BatchNormalization())

model.add(Conv2D(filters = 384, kernel_size = 3*3,strides = 1,padding = 'same',activation = 'relu'))
model.add(Conv2D(filters = 384, kernel_size = 3*3,strides = 1,padding = 'same',activation = 'relu'))
model.add(Conv2D(filters = 256, kernel_size = 3*3,strides = 1,padding = 'same',activation = 'relu'))
model.add(MaxPooling2D(pool_size = 3, strides = 2,padding ='valid'))


model.add(Flatten())
model.add(Dense(4096,activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(4096,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1000,activation='softmax'))
model.summary()


# ### Image :

# ![ALEXNET.jpg](attachment:ALEXNET.jpg)

# In[8]:


#Model Building
model = Sequential()
model.add(Conv2D(filters = 96, kernel_size = 11,strides = 4, activation = 'relu',input_shape = (524,524,3)))
model.add(MaxPooling2D(pool_size = 3, strides = 2))
#model.add(lrn())
# model.add(BatchNormalization())

model.add(Conv2D(filters = 256, kernel_size = 5,strides = 1, activation = 'relu'))
model.add(MaxPooling2D(pool_size = 3, strides = 2,))
# # model.add(BatchNormalization())

model.add(Conv2D(filters = 384, kernel_size = 3*3,strides = 1, activation = 'relu'))
model.add(Conv2D(filters = 384, kernel_size = 3*3,strides = 1, activation = 'relu'))


model.add(Conv2D(filters = 256, kernel_size = 3*3,strides = 1, activation = 'relu'))
model.add(MaxPooling2D(pool_size = 3, strides = 2))


model.add(Flatten())
model.add(Dense(4096,activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(4096,activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(2,activation='softmax'))
model.summary()


# # Topic 4: Resnet

# ![RESNET_MNIST_model_plot.png](attachment:RESNET_MNIST_model_plot.png)

# In[9]:


def Residual_block(inputs):
    shortcut = inputs
    conv2 = Conv2D(filters = 64, kernel_size = 3,strides = 1, padding = 'same')(inputs)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    
    conv2 = Conv2D(filters = 64, kernel_size = 3,strides = 1, padding = 'same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    return add([shortcut,conv2]) 

def Residual_block_1(inputs):
    shortcut = inputs
    conv2 = Conv2D(filters = 64, kernel_size = 3,strides = 1, padding = 'same')(inputs)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    
    conv2 = Conv2D(filters = 64, kernel_size = 3,strides = 1, padding = 'same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    return add([shortcut,conv2]) 


# In[10]:


input_layer = Input(shape=( 32, 32, 3))

conv1 = Conv2D(64, kernel_size = 7,strides = 2, padding = 'same')(input_layer)
conv1 = BatchNormalization()(conv1)
conv1 = Activation('relu')(conv1)    
maxpool1 = MaxPooling2D(pool_size = 2, strides = 2,padding ='same')(conv1)

net = Residual_block_1(maxpool1)
net = Residual_block(net) 
net = Residual_block(net)


net = Residual_block_1(net)
net = Residual_block(net)
net = Residual_block(net)
net = Residual_block(net) 

net = Residual_block_1(net)
net = Residual_block(net)
net = Residual_block(net)
net = Residual_block(net) 
net = Residual_block(net) 

net = Residual_block_1(net)
net = Residual_block(net)
net = Residual_block(net)

# Classifier block
Avg_pool = AveragePooling2D(pool_size=(2,2),strides=(1, 1))(net)
flatten = Flatten()(Avg_pool)
out = Dense(units=10,activation="softmax")(flatten)

model = Model(inputs=input_layer, outputs=out)
model.summary()


# ### Image :
# 

# ![Resnrt.jpg](attachment:Resnrt.jpg)

# In[11]:


def Residual_block(inputs):
    shortcut = inputs
    conv2 = Conv2D(filters = 16, kernel_size = 3,strides = 1, padding = 'same')(inputs)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    
    conv2 = Conv2D(filters = 16, kernel_size = 3,strides = 1, padding = 'same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    return add([shortcut,conv2]) 

def Residual_block_1(inputs):
    shortcut = inputs
    conv2 = Conv2D(filters = 32, kernel_size = 3,strides = 1, padding = 'same')(inputs)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    
    conv2 = Conv2D(filters = 32, kernel_size = 3,strides = 1, padding = 'same')(inputs)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    return add([shortcut,conv2]) 

def Residual_block_2(inputs):
    shortcut = inputs
    conv2 = Conv2D(filters = 64, kernel_size = 3,strides = 1, padding = 'same')(inputs)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    
    conv2 = Conv2D(filters = 64, kernel_size = 3,strides = 1, padding = 'same')(inputs)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    return add([shortcut,conv2]) 

#Model

input_layer = Input(shape=(28,28,16))

conv1 = Conv2D(filters=16, kernel_size = 3,strides = 1, padding = 'same')(input_layer)
conv1 = BatchNormalization()(conv1)
conv1 = Activation('relu')(conv1) 

net = Residual_block(conv1)

net = Residual_block(net) 
net = Residual_block(net)
net = Residual_block(net)
net = Residual_block(net)
net = Residual_block(net)
net = Residual_block(net)
net = Residual_block(net)
net = Residual_block(net)

conv1 = Conv2D(filters=32, kernel_size = 3,strides = 2, padding = 'same')(net)
conv1 = BatchNormalization()(conv1)
conv1 = Activation('relu')(conv1) 

conv1 = Conv2D(filters=32, kernel_size = 3,strides = 1, padding = 'same')(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = Activation('relu')(conv1) 

net = Residual_block_1(conv1)

net = Residual_block_1(net)
net = Residual_block_1(net)
net = Residual_block_1(net)
net = Residual_block_1(net)
net = Residual_block_1(net)
net = Residual_block_1(net)
net = Residual_block_1(net)

conv1 = Conv2D(filters=64, kernel_size = 3,strides = 2, padding = 'same')(net)
conv1 = BatchNormalization()(conv1)
conv1 = Activation('relu')(conv1) 

conv1 = Conv2D(filters=64, kernel_size = 3,strides = 1, padding = 'same')(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = Activation('relu')(conv1) 

net = Residual_block_2(conv1)

net = Residual_block_2(net)
net = Residual_block_2(net)
net = Residual_block_2(net)
net = Residual_block_2(net)
net = Residual_block_2(net)
net = Residual_block_2(net)
net = Residual_block_2(net)

Avg_pool = AveragePooling2D(pool_size=(7,7),strides=(1, 1))(net)

flatten = Flatten()(Avg_pool)

out = Dense(units=2,activation="softmax")(flatten)

model = Model(inputs=input_layer, outputs=out)
model.summary()


# ### Image :

# ![ResNet-50.jpg](attachment:ResNet-50.jpg)

# In[12]:


def ResNet50(input_shape=(224, 224, 3)):

    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')


    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    X = X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)
    
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model


# In[13]:


# Implementation of Identity Block

def identity_block(X, f, filters, stage, block):
   
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters

    X_shortcut = X
   
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X = Add()([X, X_shortcut])# SKIP Connection
    X = Activation('relu')(X)

    return X


# In[14]:


#Implementation of Convolutional Block

def convolutional_block(X, f, filters, stage, block, s=2):
   
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


# In[15]:


from keras.initializers import glorot_uniform

base_model = ResNet50(input_shape=(224, 224, 3))

headModel = base_model.output
headModel = Flatten()(headModel)
headModel=Dense(256, activation='relu', name='fc1',kernel_initializer=glorot_uniform(seed=0))(headModel)
headModel=Dense(128, activation='relu', name='fc2',kernel_initializer=glorot_uniform(seed=0))(headModel)
headModel = Dense( 1,activation='sigmoid', name='fc3',kernel_initializer=glorot_uniform(seed=0))(headModel)

model = Model(inputs=base_model.input, outputs=headModel)

model.summary()


# # Topic 5 : VGG16

# ![VGG16.png](attachment:VGG16.png)

# In[16]:


input_shape = (224, 224, 1)

model = keras.Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))        
          
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
          
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(4096))
model.add(Dense(4096))         
model.add(Dense(4096, activation="softmax"))
          
          
model.summary()
          


# ### Image : VGG16

# ![VGG16.jpg](attachment:VGG16.jpg)

# In[17]:


input_shape = (224, 224, 3)

model = keras.Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), strides=1, padding = 'same', activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), strides=1, padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=2))

model.add(Conv2D(128, kernel_size=(3, 3), strides=1, padding = 'same', activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), strides=1, padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))        
          
model.add(Conv2D(256, kernel_size=(3, 3), strides=1, padding = 'same', activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), strides=1, padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
          
model.add(Conv2D(512, kernel_size=(3, 3), strides=1, padding = 'same', activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), strides=1, padding = 'same', activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), strides=1, padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(512, kernel_size=(3, 3), strides=1, padding = 'same', activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), strides=1, padding = 'same', activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), strides=1, padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Flatten())

model.add(Dense(25088))
model.add(Dense(4096))      
model.add(Dense(4096)) 
model.add(Dense(4096, activation="softmax"))

model.summary()


# ### Image : VGG16

# ![Screenshot_20221205-100720_Drive.jpg](attachment:Screenshot_20221205-100720_Drive.jpg)

# In[18]:


input_shape = (224, 224, 3)

model = keras.Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), padding = 'same', activation='relu', input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(128, kernel_size=(3, 3), padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(256, kernel_size=(3, 3), padding = 'same', activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(512, kernel_size=(3, 3), padding = 'same', activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(512, kernel_size=(3, 3), padding = 'same', activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Flatten())

model.add(Dense(4096))     
model.add(Dropout(0.5))

model.add(Dense(4096))     
model.add(Dropout(0.5))

model.add(Dense(2, activation="softmax"))

model.summary()


# ### Image : 

# ![cover_VGG16_1600px_web-1280x640.jpg](attachment:cover_VGG16_1600px_web-1280x640.jpg)

# In[19]:


input_shape = (224, 224, 3)

model = keras.Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), padding = 'same', activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, kernel_size=(3, 3), padding = 'same', activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, kernel_size=(3, 3), padding = 'same', activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), padding = 'same', activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(512, kernel_size=(3, 3), padding = 'same', activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), padding = 'same', activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(512, kernel_size=(3, 3), padding = 'same', activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), padding = 'same', activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(4096))     
model.add(Dropout(0.5))

model.add(Dense(4096))     
model.add(Dropout(0.5))

model.add(Dense(1000, activation="softmax"))

model.summary()


# # Topic 6 : Mobilenet

# ![Illustration-of-the-MobileNet-architecture-A-The-overall-MobileNet-architecture-and.png](attachment:Illustration-of-the-MobileNet-architecture-A-The-overall-MobileNet-architecture-and.png)

# In[20]:


def convolution_block(input_layer, strides, filters):
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(filters=filters, kernel_size=1, strides=1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x

input_img = Input(shape=(224, 224, 3))

x = Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), padding='same')(input_img)
x = BatchNormalization()(x)
x = ReLU()(x)
x = convolution_block(x, filters = 64, strides = 1)
x = convolution_block(x, filters = 128, strides = 2)
x = convolution_block(x, filters = 128, strides = 1)
x = convolution_block(x, filters = 256, strides = 2)
x = convolution_block(x, filters = 256, strides = 1)
x = convolution_block(x, filters = 512, strides = 2)

for i in range (5):
     x = convolution_block(x, filters = 512, strides = 1)

x = convolution_block(x, filters = 1024, strides = 2)
x = convolution_block(x, filters = 1024, strides = 1)
x = GlobalAveragePooling2D()(x)

output = Dense (10, activation = 'softmax')(x)
model = Model(inputs=input_img, outputs = output)
model.summary()


# # Topic 7 : Transfer Learning

# In[21]:


vgg_conv = applications.VGG16(weights='imagenet', include_top=False,input_shape = (224,224,3))


# In[22]:


vgg_conv.summary()


# In[23]:


#TO MAKE LAYER TRAINABLE AND NON-TRAINABLE

# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False
    
#check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)


# In[24]:


model = Sequential()
model.add(vgg_conv)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))
model.summary()


# In[25]:


vgg_conv = applications.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

layer_name = 'block4_pool'

intermediate_layer_model = Model(inputs=vgg_conv.input, outputs=vgg_conv.get_layer(layer_name).output)

intermediate_layer_model.summary()


# In[26]:


model = Sequential()

model.add(intermediate_layer_model)

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(3, activation='softmax'))

model.summary()


# # Topic 8 : VGG19

# ![VGG-19.png](attachment:VGG-19.png)

# In[27]:


input_shape = (512, 512, 64)

model = keras.Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(4096))   
model.add(Dense(4096))     

model.add(Dense(1000, activation="softmax"))

model.summary()


# In[28]:


model = Sequential()

model.add(Conv2D(6, kernel_size=(5, 5), strides=1, activation='relu', input_shape = (32,32,6)))
model.add(AveragePooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(16, kernel_size=(5, 5), strides=1, activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2),strides=2))

model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))

model.add(Dense(10, activation="softmax"))

model.summary()


# # Topic 9 : GoogLeNet

# ![GoogLeNet.jpg](attachment:GoogLeNet.jpg)

# In[ ]:





# ### Image : GoogLeNet

# ![googlenet.webp](attachment:googlenet.webp)

# In[29]:


def Inception_block(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4):
    
    # 1st path:
    path1 = Conv2D(filters=f1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
    
    # 2nd path
    path2 = Conv2D(filters = f2_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
    path2 = Conv2D(filters = f2_conv3, kernel_size = (3,3), padding = 'same', activation = 'relu')(path2) 
    
    # 3rd path
    path3 = Conv2D(filters = f3_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
    path3 = Conv2D(filters = f3_conv5, kernel_size = (5,5), padding = 'same', activation = 'relu')(path3)
    
    # 4th path
    path4 = MaxPooling2D((3,3), strides= (1,1), padding = 'same')(input_layer)
    path4 = Conv2D(filters = f4, kernel_size = (1,1), padding = 'same', activation = 'relu')(path4)
    
    output_layer = concatenate([path1, path2, path3, path4], axis = -1)
    
    return output_layer


# Input: 
  # - f1: number of filters of the 1x1 convolutional layer in the first path
  # - f2_conv1, f2_conv3 are number of filters corresponding to the 1x1 and 3x3 convolutional layers in the second path
  # - f3_conv1, f3_conv5 are the number of filters corresponding to the 1x1 and 5x5  convolutional layer in the third path
  # - f4: number of filters of the 1x1 convolutional layer in the fourth path


# In[30]:


def GoogLeNet():
    # input layer
    input_layer = Input(shape = (224, 224, 3))
    
    # convolutional layer: filters = 64, kernel_size = (7,7), strides = 2
    X = Conv2D(filters = 64, kernel_size = (7,7), strides = 2, padding = 'valid', activation = 'relu')(input_layer)
    
    # max-pooling layer: pool_size = (3,3), strides = 2
    X = MaxPooling2D(pool_size = (3,3), strides = 2)(X)
    
    # convolutional layer: filters = 64, strides = 1
    X = Conv2D(filters = 64, kernel_size = (1,1), strides = 1, padding = 'same', activation = 'relu')(X)
    
    # convolutional layer: filters = 192, kernel_size = (3,3)
    X = Conv2D(filters = 192, kernel_size = (3,3), padding = 'same', activation = 'relu')(X)
    
    # max-pooling layer: pool_size = (3,3), strides = 2
    X = MaxPooling2D(pool_size= (3,3), strides = 2)(X)
    
    # 1st Inception block
    X = Inception_block(X, f1 = 64, f2_conv1 = 96, f2_conv3 = 128, f3_conv1 = 16, f3_conv5 = 32, f4 = 32)
    
    # 2nd Inception block
    X = Inception_block(X, f1 = 128, f2_conv1 = 128, f2_conv3 = 192, f3_conv1 = 32, f3_conv5 = 96, f4 = 64)
    
    # max-pooling layer: pool_size = (3,3), strides = 2
    X = MaxPooling2D(pool_size= (3,3), strides = 2)(X)
    
    # 3rd Inception block
    X = Inception_block(X, f1 = 192, f2_conv1 = 96, f2_conv3 = 208, f3_conv1 = 16, f3_conv5 = 48, f4 = 64)
    
    # Extra network 1:
    X1 = AveragePooling2D(pool_size = (5,5), strides = 3)(X)
    X1 = Conv2D(filters = 128, kernel_size = (1,1), padding = 'same', activation = 'relu')(X1)
    X1 = Flatten()(X1)
    X1 = Dense(1024, activation = 'relu')(X1)
    X1 = Dropout(0.7)(X1)
    X1 = Dense(5, activation = 'softmax')(X1)
    
    # 4th Inception block
    X = Inception_block(X, f1 = 160, f2_conv1 = 112, f2_conv3 = 224, f3_conv1 = 24, f3_conv5 = 64, f4 = 64)
    
    # 5th Inception block
    X = Inception_block(X, f1 = 128, f2_conv1 = 128, f2_conv3 = 256, f3_conv1 = 24, f3_conv5 = 64, f4 = 64)
    
    # 6th Inception block
    X = Inception_block(X, f1 = 112, f2_conv1 = 144, f2_conv3 = 288, f3_conv1 = 32, f3_conv5 = 64, f4 = 64)
    
    # Extra network 2:
    X2 = AveragePooling2D(pool_size = (5,5), strides = 3)(X)
    X2 = Conv2D(filters = 128, kernel_size = (1,1), padding = 'same', activation = 'relu')(X2)
    X2 = Flatten()(X2)
    X2 = Dense(1024, activation = 'relu')(X2)
    X2 = Dropout(0.7)(X2)
    X2 = Dense(1000, activation = 'softmax')(X2)
  
  
    # 7th Inception block
    X = Inception_block(X, f1 = 256, f2_conv1 = 160, f2_conv3 = 320, f3_conv1 = 32, f3_conv5 = 128, f4 = 128)

    # max-pooling layer: pool_size = (3,3), strides = 2
    X = MaxPooling2D(pool_size = (3,3), strides = 2)(X)

  # 8th Inception block
    X = Inception_block(X, f1 = 256, f2_conv1 = 160, f2_conv3 = 320, f3_conv1 = 32, f3_conv5 = 128, f4 = 128)

  # 9th Inception block
    X = Inception_block(X, f1 = 384, f2_conv1 = 192, f2_conv3 = 384, f3_conv1 = 48, f3_conv5 = 128, f4 = 128)

  # Global Average pooling layer 
    X = GlobalAveragePooling2D(name = 'GAPL')(X)

  # Dropoutlayer 
    X = Dropout(0.4)(X)

  # output layer 
    X = Dense(1000, activation = 'softmax')(X)
  
  # model
    model = Model(input_layer, [X, X1, X2], name = 'GoogLeNet')

    return model


# In[31]:


model = GoogLeNet()


# In[32]:


model.summary()


# In[ ]:





# # Topic 10 : Inception

# ![inception%20v3.jpg](attachment:inception%20v3.jpg)

# In[ ]:





# ### Image :

# ![Inception%20Bn.jpg](attachment:Inception%20Bn.jpg)

# In[ ]:





# ### Image :

# ![Inception%20V3.jpg](attachment:Inception%20V3.jpg)

# In[ ]:





# In[ ]:





# # Topic 11 : Xception

# ![Xception.jpg](attachment:Xception.jpg)

# In[33]:


#import necessary libraries

import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Conv2D,Add
from tensorflow.keras.layers import SeparableConv2D,ReLU
from tensorflow.keras.layers import BatchNormalization,MaxPool2D
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras import Model
# creating the Conv-Batch Norm block

def conv_bn(x, filters, kernel_size, strides=1):
    
    x = Conv2D(filters=filters, 
               kernel_size = kernel_size, 
               strides=strides, 
               padding = 'same', 
               use_bias = False)(x)
    x = BatchNormalization()(x)
    return x
# creating separableConv-Batch Norm block

def sep_bn(x, filters, kernel_size, strides=1):
    
    x = SeparableConv2D(filters=filters, 
                        kernel_size = kernel_size, 
                        strides=strides, 
                        padding = 'same', 
                        use_bias = False)(x)
    x = BatchNormalization()(x)
    return x
# entry flow

def entry_flow(x):
    
    x = conv_bn(x, filters =32, kernel_size =3, strides=2)
    x = ReLU()(x)
    x = conv_bn(x, filters =64, kernel_size =3, strides=1)
    tensor = ReLU()(x)
    
    x = sep_bn(tensor, filters = 128, kernel_size =3)
    x = ReLU()(x)
    x = sep_bn(x, filters = 128, kernel_size =3)
    x = MaxPool2D(pool_size=3, strides=2, padding = 'same')(x)
    
    tensor = conv_bn(tensor, filters=128, kernel_size = 1,strides=2)
    x = Add()([tensor,x])
    
    x = ReLU()(x)
    x = sep_bn(x, filters =256, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters =256, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding = 'same')(x)
    
    tensor = conv_bn(tensor, filters=256, kernel_size = 1,strides=2)
    x = Add()([tensor,x])
    
    x = ReLU()(x)
    x = sep_bn(x, filters =728, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters =728, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding = 'same')(x)
    
    tensor = conv_bn(tensor, filters=728, kernel_size = 1,strides=2)
    x = Add()([tensor,x])
    return x
# middle flow

def middle_flow(tensor):
    
    for _ in range(8):
        x = ReLU()(tensor)
        x = sep_bn(x, filters = 728, kernel_size = 3)
        x = ReLU()(x)
        x = sep_bn(x, filters = 728, kernel_size = 3)
        x = ReLU()(x)
        x = sep_bn(x, filters = 728, kernel_size = 3)
        x = ReLU()(x)
        tensor = Add()([tensor,x])
        
    return tensor
# exit flow

def exit_flow(tensor):
    
    x = ReLU()(tensor)
    x = sep_bn(x, filters = 728,  kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters = 1024,  kernel_size=3)
    x = MaxPool2D(pool_size = 3, strides = 2, padding ='same')(x)
    
    tensor = conv_bn(tensor, filters =1024, kernel_size=1, strides =2)
    x = Add()([tensor,x])
    
    x = sep_bn(x, filters = 1536,  kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters = 2048,  kernel_size=3)
    x = GlobalAvgPool2D()(x)
    
    x = Dense (units = 1000, activation = 'softmax')(x)
    
    return x
# model code

input = Input(shape = (299,299,3))
x = entry_flow(input)
x = middle_flow(x)
output = exit_flow(x)

model = Model (inputs=input, outputs=output)
model.summary()


# In[34]:


# Import required packages
from keras.layers import Input, Lambda, Dense
from keras.models import Model

# Define input tensor
inputs = Input(shape=(224, 224, 3))

# Define the entry flow
x = Lambda(lambda x: x)(inputs)
x = Lambda(lambda x: x)(x)

# Define the middle flow
for i in range(8):
    x = Lambda(lambda x: x)(x)

# Define the exit flow
x = Lambda(lambda x: x)(x)
x = Dense(1024, activation='relu')(x)

# Define the model
model = Model(inputs=inputs, outputs=x)
model.summary()


# In[35]:


# Import the necessary layers and models from Keras
from keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from keras.models import Model

# Define the input layer
inputs = Input(shape=(224, 224, 3))

# Define the first block of the Xception model
x = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(inputs)
x = Conv2D(64, (3, 3), padding='same')(x)

# Define the second block of the Xception model
x = SeparableConv2D(128, (3, 3), padding='same')(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

# Define the third block of the Xception model
x = SeparableConv2D(128, (3, 3), padding='same')(x)
x = SeparableConv2D(128, (3, 3), padding='same')(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

# Define the fourth block of the Xception model
x = SeparableConv2D(256, (3, 3), padding='same')(x)
x = SeparableConv2D(256, (3, 3), padding='same')(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

x = SeparableConv2D(728, (3, 3), padding='same')(x)
x = SeparableConv2D(728, (3, 3), padding='same')(x)
x = SeparableConv2D(728, (3, 3), padding='same')(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)


model = Model(inputs=inputs, outputs=x)
model.summary()


# In[36]:


def xception(input_shape, n_classes):
  
  def conv_bn(x, f, k, s=1, p='same'):
    x = Conv2D(f, k, strides=s, padding=p, use_bias=False)(x)
    x = BatchNormalization()(x)
    return x
  
  
  def sep_bn(x, f, k, s=1, p='same'):
    x = SeparableConv2D(f, k, strides=s, padding=p, use_bias=False)(x)
    x = BatchNormalization()(x)
    return x
  
  
  def entry_flow(x):
    x = conv_bn(x, 32, 3, 2)
    x = ReLU()(x)
    x = conv_bn(x, 64, 3)
    tensor = ReLU()(x)
    
    x = sep_bn(tensor, 128, 3)
    x = ReLU()(x)
    x = sep_bn(x, 128, 3)
    x = MaxPool2D(3, strides=2, padding='same')(x)
    
    tensor = conv_bn(tensor, 128, 1, 2)
    
    x = Add()([tensor, x])
    x = ReLU()(x)
    x = sep_bn(x, 256, 3)
    x = ReLU()(x)
    x = sep_bn(x, 256, 3)
    x = MaxPool2D(3, strides=2, padding='same')(x)
    
    tensor = conv_bn(tensor, 256, 1, 2)
    
    x = Add()([tensor, x])
    x = ReLU()(x)
    x = sep_bn(x, 728, 3)
    x = ReLU()(x)
    x = sep_bn(x, 728, 3)
    x = MaxPool2D(3, strides=2, padding='same')(x)
    
    tensor = conv_bn(tensor, 728, 1, 2)
    x = Add()([tensor, x])
    
    return x
  
  
  def middle_flow(tensor):
    for _ in range(8):
        x = ReLU()(tensor)
        x = sep_bn(x, 728, 3)
        x = ReLU()(x)
        x = sep_bn(x, 728, 3)
        x = ReLU()(x)
        x = sep_bn(x, 728, 3)

        tensor = Add()([tensor, x])
    
    return tensor
  
  
  def exit_flow(tensor):
    x = ReLU()(tensor)
    x = sep_bn(x, 728, 3)
    x = ReLU()(x)
    x = sep_bn(x, 1024, 3)
    x = MaxPool2D(3, strides=2, padding='same')(x)
    
    tensor = conv_bn(tensor, 1024, 1, 2)
    
    x = Add()([tensor, x])
    x = sep_bn(x, 1536, 3)
    x = ReLU()(x)
    x = sep_bn(x, 2048, 3)
    x = ReLU()(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(n_classes, activation='softmax')(x)
  
    return x
  
  
  input = Input(input_shape)
  
  x = entry_flow(input)
  x = middle_flow(x)
  output = exit_flow(x)
  
  model = Model(input, output)
  
  return model


# In[37]:


get_ipython().run_cell_magic('capture', '', 'import keras\nimport keras.backend as K\nfrom keras.models import Model\nfrom keras.layers import Input, Dense, Conv2D, Conv3D, DepthwiseConv2D, SeparableConv2D, Conv3DTranspose\nfrom keras.layers import Flatten, MaxPool2D, AvgPool2D, GlobalAvgPool2D, UpSampling2D, BatchNormalization\nfrom keras.layers import Concatenate, Add, Dropout, ReLU, Lambda, Activation, LeakyReLU, PReLU\n\nfrom IPython.display import SVG\nfrom keras.utils.vis_utils import model_to_dot\n\nfrom time import time\nimport numpy as np\n\ninput_shape = 224, 224, 3\nn_classes = 1000\n\nK.clear_session()\nmodel = xception(input_shape, n_classes)\nmodel.summary()')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




