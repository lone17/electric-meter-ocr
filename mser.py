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
from Levenshtein import distance

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

def get_cropped_images(regions, image, target_size=(32, 32), trim=False, plot_debug=False):
    region_images = []
    
    for i, (x, y, w, h) in enumerate(regions):
        cropped_image = image[y:y+h, x:x+w]
        # print(x,y,w,h)
        # plt.subplot('131')
        # plt.imshow(np.sort(cropped_image, axis=1))
        if h / w > 1.5 and trim:
            if plot_debug:
                plt.subplot('132')
                plt.imshow(cropped_image)
            trim_row = trim_row_index(cropped_image)
            if trim_row < h / 4:
                regions[i][1] += trim_row
                cropped_image = cropped_image[trim_row:]
                if plot_debug:
                    plt.subplot('133')
                    plt.title(trim_row)
                    plt.imshow(cropped_image)
            if plot_debug:
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
        # images = np.array([global_hist_equalize(img) for img in images], dtype='float')
        images = np.array([convert_to_gray(img) for img in images], dtype='float')
    
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
    
    # print(start, longest, 'asdad')
    return start + longest // 2

def filt_boxes(boxes, image):
    keep_indices = []
    image_h, image_w = image.shape[:2]
    image_area = image_h * image_w
    for i, (x, y, w, h) in enumerate(boxes):
        # too small
        if image_w / w > 15:
            continue
        if image_h / h < 0.2:
            continue
        if image_area / (w * h) > 32:
            continue
        # too big
        if image_area / (w * h) < 5:
            continue
        # weird shape
        if w / h > 1.5 or h / w > 3:
            continue
        keep_indices.append(i)
    
    return boxes[keep_indices]

def get_rotate_angle(img, max_degree=10, plot_debug=False):
    img = bilateral_blur(img.copy(), 9, 50, 50)
    img = sharpen(img)
    # plt.imshow(img)
    # plt.show()
    gray_img = clahe(img, clipLimit=2.0, tileGridSize=(21, 31))
    # gray_img = convert_to_gray(img)
    edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
    if plot_debug:
        plt.subplot('311')
        plt.imshow(edges)
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    
    display_img = gray_img.copy()
    
    try:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0][0], line[0][1], line[0][2], line[0][3]
            if x2 - x1 > 30 and np.abs((y2 - y1) / (x2 - x1)) < np.tan(np.radians(max_degree)):
                angles.append(np.arctan((y2 - y1) / (x2 - x1)))
                if plot_debug:
                    cv2.line(display_img, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)
        
        if len(angles) > 0:
            rotate_angle = np.mean(angles)
            rotate_angle = np.degrees(rotate_angle)
        else:
            rotate_angle = 0
    except Exception as e:
        rotate_angle = 0
    
    if plot_debug:
        plt.subplot('312')
        plt.imshow(display_img)
        print(rotate_angle)
        display_img = imutils.rotate(display_img, rotate_angle)
        plt.subplot('313')
        plt.imshow(display_img)
        plt.show()
    
    return rotate_angle

def get_red_blob_bounding_box(img, plot_debug=False):
    tmp = gamma_correct(img.copy())
    
    tmp = tmp[..., 2] - 0.5 * (tmp[..., 0] + tmp[..., 1])
    tmp -= np.min(tmp)
    tmp = tmp / np.max(tmp) * 255
    tmp = tmp.astype('uint8')
    
    pixel_values = np.sort(tmp.ravel())
    threshold = pixel_values[int(0.95 * len(pixel_values))]
    tmp = tmp * (tmp > threshold)
    
    tmp[:, :int(0.75 * tmp.shape[1])] = 0
    tmp[:int(0.1 * tmp.shape[0]), :] = 0
    tmp[-int(0.1 * tmp.shape[0]):, :] = 0
    
    _, contours, _ = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blob = max(contours, key=lambda el: cv2.contourArea(el))
    poly = cv2.approxPolyDP(blob, 3, True)
    x, y, w, h = cv2.boundingRect(poly)
    
    if plot_debug:
        cv2.rectangle(tmp, (x-5, y-5), (x+w+5, y+h+5), (255, 255, 255))
        plt.imshow(tmp)
        plt.show()
    
    return (x, y, w, h)

clf_model = get_classifier_model(num_classes=2, num_filters=32)
clf_model.load_weights('clf_weights.h5')
# clf_model = load_model('detector_model.hdf5')

rcn_model = get_classifier_model(num_classes=10, num_filters=64)
rcn_model.load_weights('rcn_weights.h5')

plt.rcParams["figure.figsize"] = [9, 9]
loss = []
plot_debug = True
# for img_path in list(paths.list_images(r'D:\Google Drive\image processing\image_cropped'))[:]:
#     origin_img = cv2.imread(img_path) 
#     # origin_img = resize_to_prefered_height(origin_img, prefered_height=240)
#     label = '00000'
from load_data import test_data
file_list = os.listdir(r'D:\Google Drive\image processing\image_cropped')
for img_idx, (origin_img, label) in enumerate(test_data[:]):
    label = ''.join(label)[:5]

    img = origin_img.copy()

    rotate_angle = get_rotate_angle(img, max_degree=10)
    
    img = clahe(img, clipLimit=3.0, tileGridSize=(10, 17))
    img = imutils.rotate(img, rotate_angle)
    origin_img = imutils.rotate(origin_img, rotate_angle)
    # img = img[:, :int(0.8 * img.shape[1])]
    # origin_img = origin_img[:, :int(0.8 * origin_img.shape[1])]
    
    # img = global_hist_equalize(img)
    # img = thresh(img)
    # plt.subplot('411')
    # plt.imshow(img)

    processed_img = clahe(img, clipLimit=3.0, tileGridSize=(10, 17))
    boxes = get_region_candidates(processed_img)
    boxes = filt_boxes(boxes, img)

    if plot_debug:
        display_img = clahe(img, clipLimit=3.0, tileGridSize=(10, 17))
        for x, y, w, h in boxes:
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        plt.subplot('412')
        plt.imshow(display_img)

    # print(img.shape)
    # print(global_hist_equalize(img).shape)
    region_images, regions = get_cropped_images(boxes, img, trim=False)

    processed_images = preprocess_images(region_images, mode='clf')
    probs = clf_model.predict_proba(processed_images, verbose=0)[:, 1]
    
    # for i, (_, _, w, h) in enumerate(boxes):
    #     # if h / w > 1.6 and h / w < 1.7:
    #     #     probs[i] += 0.1
    #     if h / w >= 1.75:
    #         probs[i] -= 0.1

    mask = probs > 0.5
    boxes = boxes[mask]
    region_images = region_images[mask]
    probs = probs[mask]
    
    if plot_debug:
        display_img = origin_img.copy()
        for i, (x, y, w, h) in enumerate(boxes):
            display_img[y:y+h, x:x+w] = cv2.cvtColor(convert_to_gray(img[y:y+h, x:x+w])[:,:,0], cv2.COLOR_GRAY2BGR)
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(display_img, str(probs[i]), (x+5, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=3)
        plt.subplot('413')
        plt.imshow(display_img)

    # boxes, probs = merge_boxes(boxes, probs)
    # # boxes, probs = merge_boxes(boxes, probs)
    # # print(boxes)
    # sort_indices = np.argsort(boxes[:, 0])
    # boxes = np.array([boxes[i] for i in sort_indices])
    # # print(boxes)
    # region_images, regions = get_cropped_images(boxes, img, trim=True)
    
    indices = non_max_suppression(boxes, probs, 0.1)
    boxes = boxes[indices]
    region_images = region_images[indices]
    # boxes = filt_boxes(boxes, img)
    sort_indices = np.argsort(boxes[:, 0])
    boxes = np.array([boxes[i] for i in sort_indices])
    region_images = np.array([region_images[i] for i in sort_indices])
    # region_images, regions = get_cropped_images(boxes, img, trim=False)
    
    if len(region_images) > 0:
        processed_images = preprocess_images(region_images, mode='rcn')
        # for i, image in enumerate(processed_images):
        #     plt.subplot(1, len(processed_images), i+1)
        #     plt.imshow(np.sort(image[:,:,0], axis=1))
        # plt.show()
        probs = rcn_model.predict_proba(processed_images)
        preds = probs.argmax(axis=-1)

        red_blob = get_red_blob_bounding_box(origin_img.copy())
        mean_w = np.mean([w for x, y, w, h in boxes] + [red_blob[2]])
        right_most = max(x + mean_w / 2, 0.8 * origin_img.shape[1])
        left_most = np.min([w for x, y, w, h in boxes]) - mean_w / 4
        width = right_most - left_most + 1
        # keep = []
        # for i, (x, y, w, h) in enumerate(boxes):
        #     if (x + w / 2) < right_most:
        #         keep.append(i)
        # region_images = region_images[keep]
        
        prediction = [0, 0, 5, 5, 5]
        section_area = [0 for i in range(5)]
        for i, (x, y, w, h) in enumerate(boxes):
            section_idx = int((x + w / 2 - left_most) / (width / 5))
            if section_idx > 4:
                continue
            if w * h > section_area[section_idx]:
                prediction[section_idx] = preds[i]
                section_area[section_idx] = w * h
        
        if prediction[0] in [6, 8]:
            prediction[0] = 0
        if prediction[1] in [6, 8]:
            prediction[1] = 0
            
        prediction = ''.join([str(i) for i in prediction])
        
        if plot_debug:
            display_img = origin_img.copy()
            img_h, img_w = display_img.shape[:2]
            for i in range(0, 6):
                x = int(left_most + width * i / 5)
                cv2.line(display_img, (x, 0), (x, img_h), (255, 0, 0), 3)
            for i, (x, y, w, h) in enumerate(boxes):
                # print(x,y,w,h)
                display_img[y:y+h, x:x+w] = cv2.cvtColor(convert_to_gray(img[y:y+h, x:x+w])[:,:,0], cv2.COLOR_GRAY2BGR)
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(display_img, str(preds[i]), (x+5, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=3)
            plt.subplot('414')
            plt.imshow(display_img)
        
        if plot_debug:
            plt.subplot('411')
            plt.imshow(origin_img)
            plt.title(label + ' => ' + prediction)
        
        loss.append(distance(prediction, label) / max(len(prediction), len(label)))
    else:
        loss.append(1)
    
    print(label + ' => ' + prediction)
    
    if plot_debug:
        # cv2.imshow('', display_img)
        plt.savefig('debug_images/' + file_list[img_idx])
        plt.show()
    
print(np.mean(loss))
