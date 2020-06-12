# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 17:40:46 2020

@author: Venkat
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(16, (3,3), input_shape=(160,160,3), activation=tf.nn.relu))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation=tf.nn.relu))
model.add(keras.layers.Dense(1,activation=tf.nn.sigmoid))
model.summary()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['acc'])

traindatapath = 'dataset/training_set'
testdatapath = 'dataset/test_set'
datatrain = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 30,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)
datatest = ImageDataGenerator(
    rescale = 1./255)

traindata = datatrain.flow_from_directory(
    traindatapath,
    target_size=(160,160),
    class_mode = 'binary',
    batch_size = 32)

testdata = datatest.flow_from_directory(
    testdatapath,
    target_size = (160,160),
    batch_size = 16,
    class_mode = 'binary')

model.fit_generator(traindata,
                    steps_per_epoch = 150,
                    epochs = 5,
                    validation_data=testdata,
                    validation_steps = 30,
                    verbose = 1)
model.save('facemask.h5')
