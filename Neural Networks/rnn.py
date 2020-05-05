# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:13:23 2020

@author: LukaszMalucha
"""

import tensorflow as tf
from tensorflow.keras.datasets import imdb


# Dataset Parameters
number_of_words = 20000
max_len = 100

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=number_of_words)

# Pad word sequences to be the same length
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, max_len)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)

# Initialize model
model = tf.keras.models.Sequential()

# Create vector representation of words
model.add(tf.keras.layers.Embedding(input_dim=number_of_words, output_dim=128, input_shape=(X_train.shape[1],)))

# Dropout
model.add(tf.keras.layers.Dropout(0.2))

# LSTM to understand relations between elements in word sequence
model.add(tf.keras.layers.LSTM(units=128, activation="tanh"))


# Output. Sigmoid for 0-1
model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

model.fit(X_train, y_train, epochs=3, batch_size=128)