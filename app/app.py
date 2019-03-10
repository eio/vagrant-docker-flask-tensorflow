# server/app.py
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, redirect, jsonify, render_template
import os
import base64
import numpy as np
# import matplotlib.pyplot as plt
import redis
import time

# declare constants
HOST = '0.0.0.0'
PORT = 8081
REDIS_PORT = 6379
CLASS_NAMES = ['0','1','2','3','4','5','6','7','8','9']
HERE = os.path.dirname(os.path.abspath(__file__))

# initialize flask application
app = Flask(__name__)

# initialize redis store
cache = redis.Redis(host='redis', port=REDIS_PORT)

def get_hit_count():
    retries = 5
    while True:
        try:
            return cache.incr('hits')
        except redis.exceptions.ConnectionError as exc:
            if retries == 0:
                raise exc
            retries -= 1
            time.sleep(0.5)

def predict_digit(mnist_model_filepath):
    # get input data
    data = request.get_json()
    # decode the base64 encoded image
    decoded = base64.b64decode(data['b64img'])
    # create 1D numpy array from binary data string
    img = np.frombuffer(decoded, dtype=np.uint8)
    # reshape 1D array into 28x28 matrix
    img = img.reshape(28,28)
    # add image data to a batch where it's the only member
    img_batch = (np.expand_dims(img,0))
    # load our pre-trained MNIST model
    model = keras.models.load_model(mnist_model_filepath)
    # make prediction
    predictions_single = model.predict(img_batch)
    prediction = predictions_single[0]
    predicted = np.argmax(prediction)
    percent_confident = 100*np.max(prediction)
    confidence = '{:2.0f}%'.format(percent_confident)
    # # display test image and prediction
    # plt.figure()
    # plt.imshow(img, cmap='gray')
    # plt.xlabel("Prediction: {} (Confidence: {})".format(CLASS_NAMES[predicted], confidence))
    # plt.show()
    return {
        'prediction': CLASS_NAMES[predicted],
        'confidence': confidence
    }

@app.route('/')
def hello():
    count = get_hit_count()
    print('Hi my log I have been seen {} times.'.format(count))
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def classify():
    mnist_model_filepath_A = '{}/../models/10epochs_mnist_model.h5'.format(HERE)
    mnist_model_filepath_B = '{}/../models/100epochs_mnist_model.h5'.format(HERE)
    mnist_model_filepath_C = '{}/../models/1000epochs_mnist_model.h5'.format(HERE)
    return jsonify({
        '10_epochs': predict_digit(mnist_model_filepath_A),
        '100_epochs': predict_digit(mnist_model_filepath_B),
        '1000_epochs': predict_digit(mnist_model_filepath_C)
    })

if __name__ == '__main__':
    # run web server
    app.run(
        host=HOST,
        port=PORT,
        # automatic reloading enabled
        debug=True
    )
