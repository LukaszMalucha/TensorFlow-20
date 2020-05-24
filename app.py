# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:45:50 2020

@author: LukaszMalucha
"""

import os 
import requests
import numpy as np
import tensorflow as tf

from imageio import imwrite, imread
from flask import Flask, request, jsonify


# LOAD MODEL STRUCTURE
with open("fashion_model_flask.json", "r") as f:
    model_json = f.read()
    
    
model = tf.keras.models.model_from_json(model_json)

# LOAD MODEL WEIGHTS
model.load_weights("fashion_model_flask.h5")



# FLASK API
app = Flask(__name__)

@app.route("/api/v1/<string:img_name>", methods=["POST"])
def classify_image(img_name):
    
    upload_dir = "uploads/"
    
    image = imread(upload_dir + img_name)
    
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    
    prediction = model.predict([image.reshape(1, 28*28)])

    return jsonify({"object_detected":classes[np.argmax(prediction[0])]})



app.run(port=5000, debug=True)