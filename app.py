from __future__ import division, print_function
import os
import glob
import numpy as np
from flask_cors import CORS,cross_origin
from tensorflow.keras.applications.resnet50 import ResNet50

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
CORS(app)

model = ResNet50(weights='imagenet')
print('Model loaded.......')


def deleteFolder(path):
    files = glob.glob(path+'/*')
    for f in files:
        os.remove(f)


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
@cross_origin()
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        path = basepath +'\\uploads'
        print(path)
        # Make prediction
        preds = model_predict(file_path, model)

        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        deleteFolder(path)
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

