# -*- coding: utf-8 -*-
"""
Created on Mon May  4 10:13:17 2020

@author: LukaszMalucha
"""

import tensorflow as tf
import numpy as np


tf.__version__



tensor_20 = tf.constant([[23,4], [32, 51]])

tensor_20.shape


# Get only values of constant without need of a session
tensor_20.numpy()


# Converting numpy array to TF tensor

numpy_tensor = np.array([[23,4], [32,51]])
tensor_from_numpy = tf.constant(numpy_tensor)
tensor_from_numpy


# Defining variable

tf2_variable = tf.Variable([[1.,2.,3.], [4.,5.,6.]])
tf2_variable

tf2_variable.numpy()

# Change specific value in variable
tf2_variable[0,2].assign(100)
tf2_variable




# Tensor Operations

tensor = tf.constant([[1,2], [3, 4]])
tensor

tensor + 2

tensor * 5

np.square(tensor)

np.sqrt(tensor)

# Dot Product (Matrix Product)
np.dot(tensor, tensor_20)


# STRING (For deep NLP)
tf_string = tf.constant("Tensorflow")
tf_string

# Get length of a string
tf.strings.length(tf_string)

# unicode decode
tf.strings.unicode_decode(tf_string, "UTF8")



# Storring arrays of strings
tf_string_array = tf.constant(["Tensorflow", "Pytorch", "Keras"])

for string in tf_string_array:
    print(string)





































