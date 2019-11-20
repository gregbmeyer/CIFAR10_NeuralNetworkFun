# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 23:34:28 2019
reference: https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
"""

#6733 - Homework 7 - Greg Meyer

import time
start = time.time()
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from matplotlib import pyplot as mplt
end = time.time()
print("Import Time:", (end - start))

start = time.time()
batch = 20
predictedClasses = 10
epochs = 8
num_predictions = 10
(Xtrain, ytrain), (Xtest, ytest) = cifar10.load_data()
print('Xtrain shape:', Xtrain.shape)
print(Xtrain.shape[0], 'train samples')
print(Xtest.shape[0], 'test samples')

mplt.imshow(Xtrain[0])
mplt.show()
mplt.imshow(Xtrain[1])
mplt.show()
mplt.imshow(Xtrain[2])
mplt.show()
mplt.imshow(Xtrain[3])
mplt.show()
mplt.imshow(Xtrain[4])
mplt.show()

x_train = Xtrain.astype('float32')/32
x_test = Xtest.astype('float32')/32
ytrain = keras.utils.to_categorical(ytrain, predictedClasses)
ytest = keras.utils.to_categorical(ytest, predictedClasses)
end = time.time()
print("Data Train/Test Time:", (end - start))

start = time.time()
cifar10Model = Sequential()
cifar10Model.add(Conv2D(32, (5, 5), padding='valid',input_shape=x_train.shape[1:]))
cifar10Model.add(Activation('relu'))
cifar10Model.add(MaxPooling2D(pool_size=(2, 2)))
cifar10Model.add(Conv2D(64, (5, 5)))
cifar10Model.add(Activation('relu'))
cifar10Model.add(MaxPooling2D(pool_size=(2, 2)))
print("Conv2D/Pooling/Activation portion of Model Summary:", cifar10Model.summary())
cifar10Model.add(Dropout(0.3))
cifar10Model.add(Flatten())
cifar10Model.add(Dense(32))
cifar10Model.add(Activation('relu'))
cifar10Model.add(Dropout(0.6)) 
cifar10Model.add(Dense(predictedClasses)) # output dimension should match the number of things to predict
cifar10Model.add(Activation('softmax'))
opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6) # decreasing de cay rate seems to have helped some 1e-6 to 1e-5
#Categorical crossentropy because predicting many categories, not binary
cifar10Model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
end = time.time()
print("Complete Model Summary:", cifar10Model.summary())
print("Model compile time:", (end - start))

start = time.time()
#ImageDataGenerator produces additional transformed images from input images to enhance the training set
datagen = ImageDataGenerator(featurewise_center=False,samplewise_center=False, featurewise_std_normalization=False, 
        samplewise_std_normalization=False,zca_whitening=False,
        rotation_range=10,width_shift_range=0.1, height_shift_range=0.1, 
        shear_range=0.1,zoom_range=0.1,channel_shift_range=0.,fill_mode='nearest',cval=0., 
        horizontal_flip=True,vertical_flip=False,  
        rescale=None,preprocessing_function=None,data_format=None,validation_split=0.1) 
datagen.fit(Xtrain)
cifar10Model.fit_generator(datagen.flow(Xtrain, ytrain,batch_size=batch_size),epochs=epochs, steps_per_epoch = 1, validation_data=(Xtest, ytest),workers=4) #Quadcore cpu
cifar10Model.fit(Xtrain, ytrain,batch_size=batch_size,epochs=epochs,validation_data=(Xtest, ytest),shuffle=True)
end = time.time()
print("Model fit and data generator fit time:", (end - start))

start = time.time()
scores = cifar10Model.evaluate(Xtest, ytest, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
end = time.time()
print("Model Evaluation Time:", (end - start))