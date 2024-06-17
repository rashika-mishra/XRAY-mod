# %%

import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# %%
from glob import glob
train_image_path='/Users/anik_singhal/Desktop/Projects/makathon/dataset/chest_xray/train'
train_image=glob(train_image_path+'//.jp*g')
len(train_image)

# %%
test_image_path='/Users/anik_singhal/Desktop/Projects/makathon/dataset/chest_xray/test'
test_image=glob(test_image_path+'//.jp*g')
len(test_image)

# %%
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_data_generator=ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)


# %%
test_data_generator=ImageDataGenerator(rescale=1./255,shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

# %%
train_data=train_data_generator.flow_from_directory(train_image_path,
                                                   target_size=(64,64),
                                                   batch_size=32,
                                                   class_mode='binary')
test_data=test_data_generator.flow_from_directory(test_image_path,
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode='binary')

# %% [markdown]
# # training CNN model

# %%
from tensorflow.keras.layers import  Input, Conv2D, Dense, Flatten, Dropout, MaxPool2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model

input_layer=Input(shape=(64,64,3))
conv1=Conv2D(16, (3,3), strides=1, activation='relu')(input_layer)
maxpool1=MaxPool2D(2,2)(conv1)
conv2=Conv2D(32, (3,3), strides=1, activation='relu')(maxpool1)
maxpool2=MaxPool2D(2,2)(conv2)
conv3=Conv2D(16, (3,3), strides=1, activation='relu')(maxpool2)
maxpool3=MaxPool2D(2,2)(conv3)
flat=Flatten()(maxpool3)
dense1=Dense(16, activation='relu')(flat)
output_layer=Dense(1, activation='sigmoid')(dense1)

model=Model(input_layer, output_layer)

# %%
from tensorflow.keras.optimizers import Adam
adam=Adam(learning_rate=0.0001)
model.compile(optimizer=adam,
             loss='binary_crossentropy',
             metrics=['accuracy'])

# %%

# from tensorflow.keras.utils import plot_model
# plot_model(model, show_shapes=True, show_layer_names=True)

# %%

step_per_epoch=int(len(train_image)/32)-1
validation_step=int(len(test_image)/32)-1
step_per_epoch


# %%
# training the CNN model

model_history=model.fit(train_data, steps_per_epoch=step_per_epoch, epochs=25,
                       validation_data=test_data, validation_steps=validation_step)


# %%
model_history.history

# %%
model.evaluate(test_data)

# %%
# evaluating model
import matplotlib.pyplot as plt

plt.plot(model_history.history['accuracy'], label='train')
plt.plot(model_history.history['val_accuracy'], label='test')
plt.title('step vs. accuracy', size=16, c='b')
plt.xlabel('step', size=14)
plt.ylabel('accuracy', size=14)
plt.legend(loc='best')
plt.show()

## according to the graph of accuracy vs. step, this model overfits (considering the accuracy of train is higher than that of test).
# we can control several parameters to avoid this overfitting.
# CNN model's accuracy is quite high (0.85 ~ 0.92)

# %%
plt.plot(model_history.history['loss'], label='train')
plt.plot(model_history.history['val_loss'], label='test')
plt.title('step vs. loss', size=16, c='b')
plt.xlabel('step', size=14)
plt.ylabel('loss', size=14)
plt.legend(loc='best')
plt.show()

# %%
# making predictions on a single image(Normal)
from tensorflow.keras.preprocessing.image import load_img, img_to_array

file_path='/Users/anik_singhal/Desktop/Projects/makathon/dataset/chest_xray/val/NORMAL/NORMAL2-IM-1436-0001.jpeg'

# load the image
img=load_img(file_path, target_size=(64,64))
# convert to array
img=img_to_array(img)
img

# %%
img.shape

# %%
# reshape into (1, 64, 64, 3)
img=img.reshape(1,64,64,3)

# %%
img_pred=model.predict(img)
img_pred

# since 0 indicates normal case (not pneumonia), the prediction is right

# %%
# making predictions on a single image (pneumonia)
from tensorflow.keras.preprocessing.image import load_img, img_to_array

file_path2='/Users/anik_singhal/Desktop/Projects/makathon/dataset/WhatsApp Image 2024-02-24 at 16.28.48.jpeg'

# load the image
img2=load_img(file_path2, target_size=(64,64))
# convert to array
img2=img_to_array(img2)
img2

# %%
# reshape into (1, 64, 64, 3)
img2=img2.reshape(1,64,64,3)


# %%
img_pred2=model.predict(img2)
img_pred2

# since 1 indicates pneumonia case, the prediction is right

# %%
import pickle
pickle.dump(model,open('model.pkl','wb'))

# %% [markdown]
# # training VGG6 model

# %%
# vgg16 model accepts an image size of 224*224*3
# vgg16 model cosists of 5 blocks and 16 layers in total,
# each block is separated by a pooling layer
# kernel size for all Convolutional layers remain 3x3.

# from tensorflow.keras.applications.vgg16 import VGG16

# vgg16_model=VGG16()
# vgg16_model.summary()

# %%
# output from the last layer is a 1000-dimensional vector
# this is because the ImageNet dataset on which the VGG16 model is trained with 1000 classes.
# Since we are going to change the top and final layer (input_layer=(64,64) and final layer= 1 class),
# use the pretrained weight parameter, 'imagenet'
# input_shape=[64,64,3]
# myvgg16_model=VGG16(input_shape=input_shape, weights='imagenet', include_top=False)

# %%
# only train the final dense layer of our VGG16, and all other layers will not be trained.
# order not to train default VGG16 layers
# for layer in myvgg16_model.layers:
#     layer.trainable=False

# %%
# model creation for image classificatoin
# flat1=Flatten()(myvgg16_model.output)
# dense1=Dense(128, activation='relu')(flat1)
# output_layer=Dense(1, activation='sigmoid')(dense1)

# model2=Model(myvgg16_model.input, output_layer)

# %%
# # compiling model
# model2.compile(optimizer=adam,
#              loss='binary_crossentropy',
#              metrics=['accuracy'])

# %% [markdown]
# 

# %%
# from tensorflow.keras.utils import plot_model
# plot_model(model2, show_shapes=True, show_layer_names=True)

# %%
# training the VGG16 model
# model_history2=model2.fit(train_data, steps_per_epoch=step_per_epoch, epochs=20,
#                        validation_data=test_data, validation_steps=validation_step)

# %%
# model2.evaluate(test_data)

# %%
# model_history2.history

# %%
# plt.plot(pd.DataFrame(model_history2.history))
# plt.title('step vs. accuracy, loss', size=16, c='b')
# plt.xlabel('step', size=14)
# plt.ylabel('accuracy, loss', size=14)
# plt.legend(['loss','accuracy','val_loss','val_accuracy'], loc='best')
# plt.show()


## VGG16 model also slightly overfits (considering the accuracy of train is higher than that of test).
# we can control several parameters to avoid this overfitting.
# VGG16 model's accuracy is also as high as CNN model (0.85 ~ 0.92) 

# %%
# predicting on a single image

# img_pred3=model2.predict(img)
# img_pred3

# this prediction is wrong, since the img was the image of normal case, but the prediction says it is not normal

# %%
# img_pred4=model2.predict(img2)
# img_pred4

# this prediction is right, because the img2 was the image of pneumonia case

# %%