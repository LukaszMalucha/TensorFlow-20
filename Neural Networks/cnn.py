# -*- coding: utf-8 -*-
"""
Created on Mon May  4 11:08:40 2020

@author: LukaszMalucha
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10


# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load Dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


X_train = X_train / 255.0
X_test = X_test / 255.0


# Initialize model
model = tf.keras.models.Sequential()


# 32 filters 3x3, same adds column of zeros if there is odd number of columns at the end. 3 dims of input shape as images are in color
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[32,32,3]))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))

model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))

model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(units=128, activation="relu"))

model.add(tf.keras.layers.Dense(units=10, activation="softmax"))

model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["sparse_categorical_accuracy"])

model.fit(X_train, y_train, epochs=15, batch_size=128)












