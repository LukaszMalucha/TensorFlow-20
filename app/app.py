## App Utilities
import os

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify
from flask import render_template
from flask_restful import Api
from imageio import imread
from resources.image_classifier import ImageClassifier

# import env

# LOAD MODEL STRUCTURE
with open("fashion_model_flask.json", "r") as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)

# LOAD MODEL WEIGHTS
model.load_weights("fashion_model_flask.h5")

## App Settings

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
app.config['PROPAGATE_EXCEPTIONS'] = True


app.config['DEBUG'] = True
api = Api(app)



## Register Resources
api.add_resource(ImageClassifier, '/predict')



@app.route('/', methods=['GET', 'POST'])
def dashboard():
    """Main Dashboard"""
    return render_template("dashboard.html")


@app.route("/api/v1/<string:img_name>", methods=["POST"])
def classify_image(img_name):

    upload_dir = "uploads/"

    image = imread(upload_dir + img_name)

    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    prediction = model.predict([image.reshape(1, 28 * 28)])

    return jsonify({"object_detected": classes[np.argmax(prediction[0])]})



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
