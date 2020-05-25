## App Utilities
import os

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify
from flask import render_template
from flask_restful import Api
from imageio import imread
from resources.image_classifier import ImageClassify


# LOAD MODEL STRUCTURE
with open("imagenet.json", "r") as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)

# LOAD MODEL WEIGHTS
model.load_weights("imagenet.h5")

## App Settings

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
app.config['PROPAGATE_EXCEPTIONS'] = True


app.config['DEBUG'] = True
api = Api(app)
api.add_resource(ImageClassify, '/image_classify')


## Register Resources
# api.add_resource(ImageClassifier, '/predict')



@app.route('/', methods=['GET', 'POST'])
def dashboard():
    """Main Dashboard"""
    return render_template("dashboard.html")




### ERROR HANDLING


@app.errorhandler(404)
def error404(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def error500(error):
    return render_template('500.html'), 500



## APP INITIATION
if __name__ == '__main__':

    if app.config['DEBUG']:
        app.run()

    else:
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port)
