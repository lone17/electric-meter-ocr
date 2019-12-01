import cv2
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Convolution2D, Flatten, Dense, MaxPooling2D, Dropout

def resize_to_prefered_height(img, prefered_height):
    img_h, img_w = img.shape[:2]
    img = cv2.resize(img, None, fx=prefered_height/img_h, fy=prefered_height/img_h)
    
    return img

def convert_to_gray(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]
    gray = img.reshape(height, width, 1)
    
    return gray

def rotate(image, angle, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # return the rotated image
    return rotated

def get_classifier_model(input_size=32, num_classes=2, num_filters=32, dense_unit=512):
    pool_size = (2, 2)
    kernel_size = (3, 3) 
    input_shape = (input_size, input_size, 1)
    
    model = Sequential()
    model.add(Convolution2D(num_filters, kernel_size, activation='relu',
                            input_shape=input_shape))
    model.add(Convolution2D(num_filters, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size))
     
    model.add(Convolution2D(num_filters*2, kernel_size, activation='relu'))
    model.add(Convolution2D(num_filters*2, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size))
        
    model.add(Flatten())
    model.add(Dense(dense_unit, activation='relu'))
    model.add(Dropout(0.5))
    
    if True:
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])
    else:
        model.add(Dense(num_classes, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy'])
    
    return model