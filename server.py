from flask import Flask
from flask_restful import Api, Resource, reqparse
import base64
from utils import load_test_image
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)
api = Api(app)

class NN(Resource):
    def post(self, picture):
        pic = picture.replace('&', '/')
        print('pic: ', pic)

        imgdata = base64.b64decode(pic)

        with open('drawings/drawing.jpg', 'wb') as f:
            f.write(imgdata)

        X = load_test_image()
        print('X: ', X)

        # show data
        plt.imshow(X[0].reshape((28,28), order = 'F'))  
        plt.show()

        return 'processed image', 201

api.add_resource(NN, "/classify/<string:picture>")
app.run(debug=True)