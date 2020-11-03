####################################################################
# CNN for feature extraction
####################################################################

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten,Dropout
from keras.layers.core import Dense # fully-connected net
from keras import backend as K

# simple cnn model on cifar10 small images dataset

class Cnn:
  @staticmethod
  def build(width, height, depth, classes):
    # parameter: classes means the total number of classes we want to recognize
    #initialize the model
    model = Sequential()
    inputShape = (height, width, depth)

    # if we are using "channel first", update the input shape
    # in some situation like TH, use channel first
    if K.image_data_format() == "channel_first":
      inputShape = (depth, height, width)

    # first set of CONV => RELU => POOL layers => Dropout
    # conv layer will learn 32 convolution filters, each of which are 3*3
    model.add(Conv2D(32, (3, 3),padding = "same",input_shape = inputShape))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(64, (3, 3),padding = "same"))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # flattening out the volume into a set of fully-connected layer
    # first and only set of FC => RELU laters
    model.add(Flatten())
    # fully-connected layer has 512 units
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    # softmax classifier (output layer)
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    #return the constructed network architecture
    return model

####################################################################
# Auxiliary function to return the word vectors from Glove
####################################################################

import numpy as np

def get_vectors():
    vectors = {}
    f=open("glove.6B.50d.txt")
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse','ship', 'truck']
    for i in f:
        word = i.split()[0]
        if word in classes:
            vectors[word] = i.split()[1:]
    return vectors

####################################################################
# Training model
####################################################################

""" 
we will do 4 things in this file
1. Load image dataset from disk, get word vector
2. Pre-process the images if needed
3. Instantiate out convolutional neural network
4. Train out image classifier getting top N classification probability of image
5. Combine probability and word vector getting class of image
"""

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
#from cnn import Cnn
#import word_vector
import numpy as np
import cv2
import os
from keras.models import load_model

# initialize the number of epochs to train for, initial learning rate and batch size
batch_size = 32
num_classes = 8
epochs = 1
data_augmentation = False
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_zsl_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# remove ship and truck class to get train data
removed_indices = np.where(y_train!=8)[0]
x_new_train = x_train[removed_indices]
y_new_train = y_train[removed_indices]
removed_indices = np.where(y_new_train!=9)[0]
x_new_train = x_new_train[removed_indices]
y_new_train = y_new_train[removed_indices]

# remove ship and truck class to get validation data
removed_indices = np.where(y_test!=8)[0]
x_new_validation = x_test[removed_indices]
y_new_validation = y_test[removed_indices]
removed_indices = np.where(y_new_validation!=9)[0]
x_new_validation = x_new_validation[removed_indices]
y_new_validation = y_new_validation[removed_indices]

# Convert class vectors to binary class matrices, like one-hot-encoding
y_new_train = keras.utils.to_categorical(y_new_train,num_classes)
y_new_validation = keras.utils.to_categorical(y_new_validation,num_classes)

x_new_train = x_new_train.astype('float32')
x_new_validation = x_new_validation.astype('float32')
x_new_train /= 255
x_new_validation /= 255

model = Cnn.build(width=32, height=32, depth=3, classes=8)
# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_new_train, y_new_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_new_validation, y_new_validation),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_new_train, y_new_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_new_validation, y_new_validation),
                        workers=4)
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

####################################################################
# Test model
####################################################################

print("[INFO] loading network...")
model = load_model('saved_models/keras_zsl_cifar10_trained_model.h5')

indices = np.where(y_test >= 8)[0]
y_new_test = y_test[indices]
x_new_test = x_test[indices]

# combine probability got from cnn and word vector
cnn_prob = model.predict(x_new_test)

embeddings = word_vector.get_vectors()
weights = []
for i in class_labels[:-2]:
	weights.append(embeddings[i])
weights = np.array(weights,dtype=np.float32)
cnn_embedding = np.dot(cnn_prob,weights)

target_embeddings = []
for i in class_labels:
	target_embeddings.append(embeddings[i])
target_embeddings = np.array(target_embeddings, dtype=np.float32)

# use KNN to find the nearest class
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine
Y_pred_validation = []
for i in cnn_embedding:
    cos = []
    for j in target_embeddings:
    	# cosine is used to compute distance between vectors not similarity
        val = cosine(i,j)
        cos.append(val)
    # argmin returns the indices of the min values along an axis. so, we should add 1 to get the number of class
    Y_pred_validation.append(np.argmin(cos)+1)

print (accuracy_score(y_new_test, Y_pred_validation))
for i,j in zip(y_new_test, Y_pred_validation):
    print (i,j)