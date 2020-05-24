import os.path

import keras
import numpy as np
from keras.models import load_model
from skimage.io import imread
from skimage.transform import resize

# Path to saved classifier
my_path = os.path.abspath(os.path.dirname(__file__))
cat_dog_classifier = os.path.join(my_path, "../static/cat_dog_classifier/cat_dog_classifier.h5")




def image_classification(image):
    """Apply image classifier"""
    # clear Tensor session to avoid error
    keras.backend.clear_session()
    # load saved model
    image_classifier = load_model(cat_dog_classifier)
    # prepare labels
    class_labels = {0: 'Cat', 1: 'Dog'}
    # read photo & transform it into array
    img = imread(image)
    img = resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)
    if np.max(img) > 1:
        img = img / 255.0
    # predict class
    prediction = image_classifier.predict_classes(img)
    percent_values = prediction.tolist()
    # for website display
    guess = class_labels[prediction[0][0]]
    return guess



