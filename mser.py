import os

import cv2
import numpy as np
import imutils
from imutils import paths
from keras.models import load_model
from matplotlib import pyplot as plt

from utils import *
from processing import *
from models import get_classifier_model

def non_max_suppression(boxes, probs, overlap_threshold=0.3):
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes, dtype="float")
        probs = np.array(probs)
     
        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = x1 + boxes[:, 2]
        y2 = y1 + boxes[:, 3]
     
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(probs)
        # keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the index value to the list of
            # picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
    
            # find the largest (x, y) coordinates for the start of the bounding box and the
            # smallest (x, y) coordinates for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
    
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
    
            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
    
            # delete all indexes from the index list that have overlap greater than the
            # provided overlap threshold
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))
            
        # return only the bounding boxes that were picked
        return pick
    
def merge_boxes(boxes, probs, iou_threshold=0.2):
    if len(boxes) <= 5:
        return boxes, probs

    boxes = np.array(boxes, dtype="float")
    probs = np.array(probs)
    
    keep_going = True
    while keep_going:
        new_boxes = []
        new_probs = []
        
        keep_going = False
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = x1 + boxes[:, 2]
        y2 = y1 + boxes[:, 3]
     
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(probs)
        # keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            highest_prob_idx = idxs[-1]

            # find the largest (x, y) coordinates for the start of the bounding box and the
            # smallest (x, y) coordinates for the end of the bounding box
            xx1 = np.maximum(x1[highest_prob_idx], x1[idxs])
            yy1 = np.maximum(y1[highest_prob_idx], y1[idxs])
            xx2 = np.minimum(x2[highest_prob_idx], x2[idxs])
            yy2 = np.minimum(y2[highest_prob_idx], y2[idxs])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of iou
            iou = (w * h) / (area[idxs] + area[highest_prob_idx] - w * h)
            
            overlap_indices = np.where(iou > iou_threshold)[0]
            
            origin_indices = idxs[overlap_indices]
            if len(overlap_indices) > 1:
                new_x = np.average(x1[origin_indices], weights=probs[origin_indices])
                new_y = np.average(y1[origin_indices], weights=probs[origin_indices])
                new_w = np.average(x2[origin_indices], weights=probs[origin_indices]) - new_x
                new_h = np.average(y2[origin_indices], weights=probs[origin_indices]) - new_y
                keep_going = True
            else:
                new_x, new_y, new_w, new_h = boxes[highest_prob_idx]

            # if new_h / new_w > 1.3:
            #     new_x -= 5
            #     new_w += 10
            # if new_h / new_w > 1.5:
            #     new_x -= 5
            #     new_w += 10
            
            new_boxes.append(np.array([new_x, new_y, new_w, new_h]))
            new_probs.append(np.mean(probs[origin_indices]))
            # delete all indexes from the index list that have iou greater than the
            # provided iou threshold
            idxs = np.delete(idxs, overlap_indices)
        
        boxes, probs = np.array(new_boxes), np.array(new_probs)
    
    for i in range(len(new_boxes)):
        x, y, w, h = new_boxes[i]
        if h / w > 1.3:
            x -= 5
            w += 10
        if h / w > 1.5:
            x -= 5
            w += 10
        if h / w > 1.6:
            y += 8
            h -= 8
        new_boxes[i] = x, y, w, h
    
    new_boxes = np.maximum(0, new_boxes)
    return np.array(new_boxes, dtype='int'), new_probs

def get_cropped_images(regions, image, target_size=(32, 32), trim=False):
    region_images = []
    
    for i, (x, y, w, h) in enumerate(regions):
        cropped_image = image[y:y+h, x:x+w]
        # print(x,y,w,h)
        plt.subplot('131')
        plt.imshow(np.sort(cropped_image, axis=1))
        if h / w > 1.5 and trim:
            plt.subplot('132')
            plt.imshow(cropped_image)
            trim_row = trim_row_index(cropped_image)
            if trim_row < h / 4:
                regions[i][1] += trim_row
                cropped_image = cropped_image[trim_row:]
                plt.subplot('133')
                plt.title(trim_row)
                plt.imshow(cropped_image)
            plt.show()
        cropped_image = cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_AREA)
        region_images.append(cropped_image)
    
    return np.array(region_images), regions
    
def get_region_candidates(img):
    # img = clahe(img, clipLimit=2.0, tileGridSize=(31, 31))
    # # img = global_hist_equalize(img)
    # # img = thresh(img)
    # plt.subplot('411')
    # plt.imshow(img)
    gray = convert_to_gray(img)
    
    mser = cv2.MSER_create(_delta=1)
    regions, _ = mser.detectRegions(gray)
    
    regions = [cv2.boundingRect(region.reshape(-1, 1, 2)) for region in regions]
    
    return np.array(regions)

def preprocess_images(images, mode):
    if mode == 'clf':
        mean = 107.524
        images = np.array([convert_to_gray(img) for img in images], dtype='float')
    elif mode == 'rcn':
        mean = 112.833
        images = np.array([global_hist_equalize(img) for img in images], dtype='float')
        
    
    images = images - mean
    
    if len(images.shape) < 4:
        images = images[..., None]
    
    return images

def trim_row_index(image):
    # if len(image.shape) > 2:
    #     image = convert_to_gray(image)[:,:,0]
    image = global_hist_equalize(image)
    
    h, w = image.shape[:2]
    row_mean = np.sort(image, axis=1)[:, -w//5:].mean(axis=1) 
    
    row_mask = row_mean > np.mean(row_mean)
    start = 0
    cnt = 0
    longest = 0
    # mean_row_mean = np.mean(row_mean)
    for i in range(len(row_mask) - 1):
        if not row_mask[i]:
            cnt += 1
        elif row_mask[i]:
            if cnt > 0:
                if cnt > longest:
                    longest = cnt
                    start = i - cnt
                cnt = 0
    
    print(start, longest, 'asdad')
    return start + longest // 2

def filt_boxes(boxes, image):
    keep_indices = []
    image_h, image_w = image.shape[:2]
    image_area = image_h * image_w
    for i, (x, y, w, h) in enumerate(boxes):
        if image_area / (w * h) > 30:
            # too small
            continue
        if image_area / (w * h) < 5:
            # too big
            continue
        if w / h > 1.5 or h / w > 3:
            # weird shape
            continue
        keep_indices.append(i)
    
    return boxes[keep_indices]

clf_model = get_classifier_model(num_classes=2, num_filters=32)
clf_model.load_weights('clf_weights.h5')
# clf_model = load_model('detector_model.hdf5')

rcn_model = get_classifier_model(num_classes=10, num_filters=64)
rcn_model.load_weights('rcn_weights.h5')

plt.rcParams["figure.figsize"] = [9, 9]
for img_path in list(paths.list_images(r'D:\Google Drive\image processing\image_cropped'))[6:]:
    origin_img = cv2.imread(img_path) 
    origin_img = resize_to_prefered_height(origin_img, prefered_height=240)

    img = origin_img.copy()
    img = clahe(img, clipLimit=2.0, tileGridSize=(21, 31))
    

    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    display_img = img.copy()
    degrees = []
    for line in lines:
        x1, y1, x2, y2 = line[0][0], line[0][1], line[0][2], line[0][3]
        if x2 - x1 > 20 and np.abs((y2 - y1) / (x2 - x1)) < np.tan(np.radians(5)):
            degrees.append(np.arctan((y2 - y1) / (x2 - x1)))
            cv2.line(display_img, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)
    rotate_angle = np.mean(degrees)
    img = imutils.rotate_bound(img, -np.degrees(rotate_angle))
    origin_img = imutils.rotate_bound(origin_img, -np.degrees(rotate_angle))
    # plt.imshow(rotated)
    plt.show()
    
    # img = global_hist_equalize(img)
    # img = thresh(img)
    # plt.subplot('411')
    # plt.imshow(img)

    boxes = get_region_candidates(img)
    boxes = filt_boxes(boxes, img)

    display_img = origin_img.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # plt.subplot('412')
    # plt.imshow(display_img)
    # plt.show()

    # print(img.shape)
    # print(global_hist_equalize(img).shape)
    region_images, regions = get_cropped_images(boxes, img)

    processed_images = preprocess_images(region_images, mode='clf')
    probs = clf_model.predict_proba(processed_images, verbose=0)[:, 1]
    
    for i, (_, _, w, h) in enumerate(boxes):
        # if h / w > 1.6 and h / w < 1.7:
        #     probs[i] += 0.1
        if h / w >= 1.7:
            probs[i] -= 0.1

    mask = probs > 0.3
    boxes = boxes[mask]
    region_images = region_images[mask]
    probs = probs[mask]
    display_img = origin_img.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(display_img, str(probs[i]), (x+5, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=3)
        
    # plt.subplot('413')
    # plt.imshow(display_img)

    boxes, probs = merge_boxes(boxes, probs)
    # boxes, probs = merge_boxes(boxes, probs)
    # print(boxes)
    sort_indices = np.argsort(boxes[:, 0])
    boxes = np.array([boxes[i] for i in sort_indices])
    # print(boxes)
    region_images, regions = get_cropped_images(boxes, img, trim=True)
    # indices = non_max_suppression(boxes, probs, 0.1)
    # boxes = boxes[indices]
    # region_images = region_images[indices]
    
    if len(region_images) > 0:
        processed_images = preprocess_images(region_images, mode='rcn')
        for i, image in enumerate(processed_images):
            plt.subplot(1, len(processed_images), i+1)
            plt.imshow(np.sort(image[:,:,0], axis=1))
        plt.show()
        probs = rcn_model.predict_proba(processed_images)
        pred = probs.argmax(axis=-1)

        display_img = origin_img.copy()
        for i, (x, y, w, h) in enumerate(boxes):
            # print(x,y,w,h)
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(display_img, str(pred[i]), (x+5, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=3)
        plt.subplot('414')
        plt.imshow(display_img)
        # plt.show()
    
        prediction = ''.join(str(i) for i in pred[:5])
        plt.subplot('413')
        plt.imshow(origin_img)
        plt.title(prediction)
        plt.show()
