"""
Created on Fri June 22 21:40:17 2022
@author: Sayed
"""

import pathlib

## For Data Manipulation
import numpy as np
import pandas as pd
## For Splitting Data
from sklearn.model_selection import train_test_split
## For Image Manipulation and Plotting
import cv2
import PIL
import seaborn as sns
import matplotlib.pyplot as plt
## For Image Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import RandomFlip,RandomZoom,RandomRotation
## For CNN Model Creation
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Flatten,Dense,Dropout,MaxPooling2D,BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
## For Metrics Purpose
from tensorflow.math import confusion_matrix
from tensorflow.keras.metrics import Precision,Recall


IMAGE_DIMS=(200,200)
LABELS=['Normal','Pneumonia']

data_dir=pathlib.Path('data/Pneumonia/chest_xray')

train_normal=pathlib.Path('data/Pneumonia/chest_xray/train/NORMAL')
train_pneumonia=pathlib.Path('data/Pneumonia/chest_xray/train/PNEUMONIA')

val_normal=pathlib.Path('data/Pneumonia/chest_xray/val/NORMAL')
val_pneumonia=pathlib.Path('data/Pneumonia/chest_xray/val/PNEUMONIA')

test_normal=pathlib.Path('data/Pneumonia/chest_xray/test/NORMAL')
test_pneumonia=pathlib.Path('data/Pneumonia/chest_xray/test/PNEUMONIA')

train_images={
    'normal':list(train_normal.glob('*.jpeg')),
    'pneumonia':list(train_pneumonia.glob('*.jpeg'))
}

val_images={
    'normal':list(val_normal.glob('*.jpeg')),
    'pneumonia':list(val_pneumonia.glob('*.jpeg'))
}

test_images={
    'normal':list(test_normal.glob('*.jpeg')),
    'pneumonia':list(test_pneumonia.glob('*.jpeg'))
}

labels_dict={
    'normal':0,
    'pneumonia':1
}



print("Train set Length",len(train_images['normal'])+len(train_images['pneumonia']))
print("Validation set Length",len(val_images['normal'])+len(val_images['pneumonia']))
print("Test set Length",len(test_images['normal'])+len(test_images['pneumonia']))

print("Count of Normal images in Train set",len(train_images['normal']))
print("Count of Pneumonia images in Train Set",len(train_images['pneumonia']))

X_train,y_train=[],[]
for label,images in train_images.items():
    for image in images:
        img=cv2.imread(str(image),0)
        img=cv2.resize(img,IMAGE_DIMS,interpolation=cv2.INTER_AREA)
        X_train.append(img)
        y_train.append(labels_dict[label])

X_val,y_val=[],[]
for label,images in val_images.items():
    for image in images:
        img=cv2.imread(str(image),0)
        img=cv2.resize(img,IMAGE_DIMS,interpolation=cv2.INTER_AREA)
        X_val.append(img)
        y_val.append(labels_dict[label])


X_test,y_test=[],[]
for label,images in train_images.items():
    for image in images:
        img=cv2.imread(str(image),0)
        img=cv2.resize(img,IMAGE_DIMS,interpolation=cv2.INTER_AREA)
        X_test.append(img)
        y_test.append(labels_dict[label])
        
        
        
        
        
        
        
X_train=np.array(X_train)/255
X_test=np.array(X_test)/255
X_val=np.array(X_val)/255

y_train=np.array(y_train)
y_test=np.array(y_test)
y_val=np.array(y_val)


X_train=np.expand_dims(X_train,-1)
X_test=np.expand_dims(X_test,-1)
X_val=np.expand_dims(X_val,-1)


print("Shape of Train Set",X_train.shape)
print("Shape of Validation Set",X_val.shape)
print("Shape of Test Set",X_test.shape)


datagen=ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)






model=Sequential([
    Conv2D(filters=16,kernel_size=6,activation='relu'),
    MaxPooling2D(),
    Conv2D(filters=16,kernel_size=6,activation='relu'),
    MaxPooling2D(),
    BatchNormalization(),
    Flatten(),
    Dropout(0.5),
    Dense(64,activation='relu'),
    Dropout(0.5),  
    Dense(1,activation='sigmoid')
  
])
precision=tf.keras.metrics.Precision(name='precision')
recall=tf.keras.metrics.Recall(name='recall')
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy',precision,recall])
COUNT_OF_1=sum(y_train)
COUNT_OF_0=len(y_train)-COUNT_OF_1
weight_0 = (len(X_train)/COUNT_OF_0)/2.0
weight_1 = (len(X_train)/COUNT_OF_1)/2.0
class_weight = {0: weight_0, 1: weight_1}
print('Weight for class 0:  ',weight_0)
print('Weight for class 1:  ',weight_1)

earlystopping=tf.keras.callbacks.EarlyStopping(
    monitor='val_recall',
    patience=5, # epochs w no improvements.
)
model.fit(
    datagen.flow(X_train,y_train,batch_size=16),
    validation_data=datagen.flow(X_val,y_val,batch_size=12),
    shuffle=True,
    epochs=50,
    callbacks=[earlystopping],
    class_weight=class_weight
)
model.evaluate(X_test,y_test)
model.save('Models/pneumonia.h5')























