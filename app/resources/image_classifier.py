from flask import Response, json
from flask_restful import Resource, reqparse
from resources.utils import image_classification
from werkzeug.datastructures import FileStorage
from imageio import imwrite, imread
import numpy as np

class ImageClassifier(Resource):


    def get(self):
        data = {
            'welcome': 'upload image to predict a class',

        }
        js = json.dumps(data)
        response = Response(js, status=200, mimetype='application/json')
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Access-Control-Allow-Origin'] = '*'

        return response


    def post(self):
        from app import model


        parse = reqparse.RequestParser()
        parse.add_argument('image', type=FileStorage, location='files')
        args = parse.parse_args()
        image = imread(args['image'])
        classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                   "Ankle boot"]
        prediction = model.predict([image.reshape(1, 28*28)])

        data = {
            "object_detected":classes[np.argmax(prediction[0])]
        }

        js = json.dumps(data)
        response = Response(js, status=200, mimetype='application/json')
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Access-Control-Allow-Origin'] = '*'

        return response



