import numpy as np
import scipy
import cv2
import tensorflow as tf
import keras
import torch
import torchvision
import sklearn
import skimage

from keras.models import load_model
from task1 import *

class Reader:

    def __init__(self, data_folder):
        self.name = "Reader"
        self.data_folder = data_folder

    # Prepare your models
    def prepare(self):
        self.clf_model = get_classifier_model(num_classes=2, num_filters=32, dense_unit=1024)
        self.clf_model.load_weights('clf_weights.h5')
        
        self.rcn_model = get_classifier_model(num_classes=10, num_filters=32, dense_unit=512)
        self.rcn_model.load_weights('recognizer_weights_32_512.h5')

    # Implement the reading process here
    def process(self, img):
        try:
            prediction = read_cropped_image(img, self.rcn_model, self.clf_model)
        except Exception as e:
            raise e
            prediction = 5000
            
        return prediction

    # Prepare your models
    def prepare_crop(self):
        print("\tInit your models here")

    # Implement the reading process here
    def crop_and_process(self, img):
        return 5000


def check_import():
    print("Python 3.6.7")
    print("Numpy = ", np.__version__)
    print("Scipy = ", scipy.__version__)
    print("Opencv = ", cv2.__version__)
    print("Tensorflow = ", tf.__version__)
    print("Keras = ", keras.__version__)
    print("pytorch = ", torch.__version__)
    print("Torch vision = ", torchvision.__version__)
    print("Scikit-learn = ", sklearn.__version__)
    print("Scikit-image = ", skimage.__version__)

if __name__=="__main__":
    check_import()

"""
Using TensorFlow backend.
Python 3.6.7
Numpy =  1.14.5
Scipy =  1.2.1
Opencv =  4.1.1
Tensorflow =  1.14.0
Keras =  2.3.0
pytorch =  1.0.1.post2
Torch vision =  0.2.2
Scikit-learn =  0.21.3
Scikit-image =  0.14.2
"""
