# -*- coding: utf-8 -*-
"""
Created on Mon May  4 11:08:40 2020

@author: LukaszMalucha
"""

import tensorflow as tf
import numpy as np
import datetime
from tensorflow.keras.datasets import fashion_mnist



############################################################ DATA PREPROCESSING ##############################################################################

# Load Dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


# Normalize images - divide each pixel in image by 255
X_train = X_train / 255.0

X_test = X_test / 255.0


# Reshaping Dataset - flattening 28x28 image to array of pixels
X_train = X_train.reshape(-1, 28*28)
X_train.shape

X_test = X_test.reshape(-1, 28*28)
X_test.shape


########################################################### BUILDING ANN       #################################################################################


# Initialize model
model = tf.keras.models.Sequential()

# Add first fully-connected hiden layer - 128 neurons, expecting 784 pixels as an input
model.add(tf.keras.layers.Dense(units=128, activation="relu", input_shape=(784, )))

# Dropout
model.add(tf.keras.layers.Dropout(0.2))

# Add first fully-connected hiden layer - 128 neurons, expecting 784 pixels as an input
model.add(tf.keras.layers.Dense(units=64, activation='relu'))


# Dropout
model.add(tf.keras.layers.Dropout(0.2))

# Add first fully-connected hiden layer - 128 neurons, expecting 784 pixels as an input
model.add(tf.keras.layers.Dense(units=64, activation='relu'))

# Output Layer for 10 categories
model.add(tf.keras.layers.Dense(units=10, activation="softmax"))



########################################################### COMPILING ANN      #################################################################################

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['sparse_categorical_accuracy'])

# Summary chart
model.summary()

model.fit(X_train, y_train, epochs=10)


# Evaluating model

test_loss, test_accuracy = model.evaluate(X_test, y_test)



########################################################### SAVE MODEL     #################################################################################


model_json = model.to_json()
with open("fashion_model.json", "w") as json_file:
    json_file.write(model_json)


model.save_weights("fashion_model.h5")


































