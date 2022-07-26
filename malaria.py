# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 22:14:58 2022

@author: Sayed
"""

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense, MaxPool2D, Conv2D, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping


datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             rotation_range=10,
                             zoom_range=0.1,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=False,
                             vertical_flip=False,
                             rescale=1/255.0, 
                             validation_split=0.2)













trainDatagen = datagen.flow_from_directory(directory='data/malariaDataSet/cell_images/',
                                           target_size=(128,128),
                                           class_mode = 'binary',
                                           batch_size = 16,
                                           subset='training')



valDatagen = datagen.flow_from_directory(directory='data/malariaDataSet/cell_images/',
                                           target_size=(128,128),
                                           class_mode = 'binary',
                                           batch_size = 16,
                                           subset='validation')

early_stop = EarlyStopping(monitor='val_loss',patience=2)






















model = Sequential()

model.add(Conv2D(32, kernel_size=3,strides=1, activation=None, input_shape=(128,128,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))

model.add(Conv2D(112, kernel_size=3,strides=1, activation=None))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(84, kernel_size=3,strides=1, activation=None))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(64, kernel_size=3,strides=1, activation=None))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(56, kernel_size=3,strides=1, activation=None))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(32, kernel_size=3,strides=1, activation=None))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(16, kernel_size=3,strides=1, activation=None))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(trainDatagen,
                   epochs =20,
                   validation_data = valDatagen,
                   callbacks=[early_stop])

model.save('Models/malaria.h5')




























