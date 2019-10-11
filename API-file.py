## Dependecies
from __future__ import division, print_function
import os
import argparse
import numpy as np

#Image 
import cv2
from sklearn.preprocessing import normalize

# Keras
import keras
from keras.layers import *
from keras.optimizers import SGD
from keras.models import load_model, Model
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from keras import backend as K

# Flask utils

from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename
from flask import Flask, redirect, url_for, request, render_template

# Config 

parser = argparse.ArgumentParser()
parser.add_argument("-w1", "--width", help="Target Image Width", type=int, default=96)
parser.add_argument("-h1", "--height", help="Target Image Height", type=int, default=96)
parser.add_argument("-c1", "--channel", help="Target Image Channel", type=int, default=3)

parser.add_argument("-p", "--path", help="Best Model Location Path", type=str, default="models/breast_cancer_detection.h5")
parser.add_argument("-s", "--save", help="Save Uploaded Image", type=bool, default=False)

parser.add_argument("--port", help="WSGIServer Port ID", type=int, default=5004)
args = parser.parse_args()

SHAPE              = (args.width, args.height, 3)
MODEL_SAVE_PATH    = args.path
SAVE_LOADED_IMAGES = args.save

# Metrics

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precisionx = precision(y_true, y_pred)
    recallx = recall(y_true, y_pred)
    return 2*((precisionx*recallx)/(precisionx+recallx+K.epsilon()))



def create_model():
   
    kernel_size = (3,3)
    pool_size= (2,2)
    first_filters = 32
    second_filters = 64
    third_filters = 128
    
    dropout_conv = 0.3
    dropout_dense = 0.3
    
    
    model = Sequential()
    model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (96, 96, 3)))
    model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
    model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
    model.add(MaxPooling2D(pool_size = pool_size)) 
    model.add(Dropout(dropout_conv))
    
    model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
    model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
    model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
    model.add(MaxPooling2D(pool_size = pool_size))
    model.add(Dropout(dropout_conv))
    
    model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
    model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
    model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
    model.add(MaxPooling2D(pool_size = pool_size))
    model.add(Dropout(dropout_conv))
    
    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(dropout_dense))

    model.add(Dense(2, activation = "softmax"))

    #model = Model(init, x)
    model.compile(Adam(lr=0.0001), loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model

model = create_model()
print(model.summary())
model.load_weights(MODEL_SAVE_PATH)
print('Model loaded. Check http://localhost:{}/'.format(args.port))


def model_predict(img_path, model):
  
    img = np.array(cv2.imread(img_path))
   
    img = cv2.resize(img, (96,96))
    print (img.shape)
     
    
    prediction = model.predict(np.expand_dims(img, axis=0), batch_size=1)
    
    return prediction

# Threshold predictions
def threshold_arr(array):
    new_arr = []
    for ix, val in enumerate(array):
        loc = np.array(val).argmax(axis=0)
        k = list(np.zeros((len(val)), dtype=np.float16))
        k[loc]=1
        new_arr.append(k)
        
    return np.array(new_arr, dtype=np.float16)


os.chdir("deploy/")
# Define a flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('breast.html')

@app.route('/predictBreastCancer', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print('here')
        pred_class = threshold_arr(preds)[0]
        
        if pred_class[0] == 1:
            result = "This scan shows  a Breast cancer  ! (" + str(preds[0][0]) + ")"
        elif pred_class[1] == 1:
           result = " This scan shows no diseases  (" + str(preds[0][1]) + ")"
           

        if not SAVE_LOADED_IMAGES:
        	os.remove(file_path)

        return result
    return None

	
if __name__ == '__main__':
    http_server = WSGIServer(('', args.port), app)
    http_server.serve_forever()
