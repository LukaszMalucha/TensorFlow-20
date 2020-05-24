# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:35:47 2020

@author: LukaszMalucha
"""

from tensorflow.keras.datasets import fashion_mnist
from imageio import imwrite

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

for i in range(5):
    imwrite(f"{i}.png", X_test[i])