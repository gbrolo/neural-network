# Flask server to feed data into classifier.
# @author: gbrolo

import base64
import numpy as np
import matplotlib.pyplot as plt

from nn import *
from config import * 
from flask import Flask
from utils import load_test_image
from flask_restful import Api, Resource, reqparse

app = Flask(__name__)
api = Api(app)

class NN(Resource):
    def post(self, picture):
        # load image from request
        imgdata = base64.b64decode(picture.replace('&', '/'))

        with open('drawings/drawing.jpg', 'wb') as f:
            f.write(imgdata)

        X = load_test_image()

        # load optimized theta arrays
        theta1_opt = np.load('theta1_opt_69-05.npy')
        theta2_opt = np.load('theta2_opt_69-05.npy')

        # make prediction and get percentages
        pred, percentages = analize_single(theta1_opt, theta2_opt, X)
        class_ids, ids_to_class = get_class_ids()
        prediction = ids_to_class[str(pred[0])]

        slice = dict()
        slice.update({ 'prediction': prediction })

        i = 0
        for percentage in percentages:
            slice.update({ ids_to_class[str(i)]: percentage })
            i += 1

        return slice, 201

# server route
api.add_resource(NN, "/classify/<string:picture>")
app.run(debug=True)