import cv2
import numpy as np

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