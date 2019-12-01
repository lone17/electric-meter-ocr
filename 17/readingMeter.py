'''
Vu Minh Hieu - 16022405
Nguyen Thanh Tung - 16020063
Vo Le Minh Tam - 16020279
Nguyen Thi Linh - 16022409
'''

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
from task2 import crop

class Reader:

    def __init__(self, data_folder):
        self.name = "Reader"
        self.data_folder = data_folder
        self.clf_model = get_classifier_model(num_classes=2, num_filters=32, dense_unit=1024)
        self.clf_model.load_weights('clf_weights.h5')
        
        self.rcn_model = get_classifier_model(num_classes=10, num_filters=32, dense_unit=512)
        self.rcn_model.load_weights('recognizer_weights_32_512.h5')

    # Prepare your models
    def prepare(self):
        pass

    # Implement the reading process here
    def process(self, img):
        try:
            prediction, boxes = read_cropped_image(img, self.rcn_model, self.clf_model)
        except Exception as e:
            import traceback
            traceback.print_exc()
            prediction = '251'
            
        return int(prediction)

    # Prepare your models
    def prepare_crop(self):
        pass

    # Implement the reading process here
    def crop_and_process(self, img):
        try:
            x, y, w, h = crop(img)
            cropped_image = cropped_image = img[y:y+h, x:x+w]
            _, boxes = read_cropped_image(cropped_image, self.rcn_model, self.clf_model)
            
            min_x = min([box[0] for box in boxes])
            mean_w = int(np.mean([box[2] for box in boxes]))
            start_x = min(max(0, min_x - mean_w // 3), 
                          max(0, int(w - 8 * mean_w)))
            
            cropped_image = cropped_image[:, start_x:]
            prediction, boxes = read_cropped_image(cropped_image, self.rcn_model, self.clf_model)
        except Exception as e:
            import traceback
            traceback.print_exc()
            prediction = '251'
        
        return int(prediction)


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
