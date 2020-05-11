from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
import numpy as np
import os
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, request

app = Flask(__name__)

mapped_value_to_food = ['Poha', 'SabudanaKhichdi', 'Samosa']


@app.route('/home', methods=['POST'])
def home():
    data = request.files['file']
    data.save(secure_filename(data.filename))
    img_path = data.filename
    new_image = load_image(img_path, True)

    with sess.as_default():
        with graph.as_default():
            predicted_food = model.predict_classes(new_image)
            predicted_probablities = model.predict(new_image)
            print(predicted_food)
            print(predicted_probablities)
            print(mapped_value_to_food[predicted_food[0]])
            resp = {"Predicted-Class": int(predicted_food[0]),
                    "Predicted-Food": mapped_value_to_food[predicted_food[0]],
                    "Probablities": [ 
                                    { 
                                        mapped_value_to_food[0] : float(predicted_probablities[0][0]),
                                        mapped_value_to_food[1] : float(predicted_probablities[0][1]),
                                        mapped_value_to_food[2] : float(predicted_probablities[0][2])
                                    }
                                ]
                    }
    return jsonify(resp), 200


def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(224, 224))
    # (height, width, channels)
    img_tensor = image.img_to_array(img)
    # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    # imshow expects values in the range [0, 1]
    img_tensor /= 255.
    return img_tensor


if __name__ == "__main__":
    sess = tf.Session()
    graph = tf.get_default_graph()
    set_session(sess)
    model = load_model('projectmodel.h5')

    app.run(host='0.0.0.0')
