import os
import json

import cv2
import numpy as np
from imutils import paths
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder

from utils import resize_to_prefered_height


enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
enc.fit(np.array(list(range(10))).reshape((-1,1)))
img_h = 240

train_data = []
val_data = []
test_data = []

labels = json.load(open('image_cropped_json.json', 'r'))['_via_img_metadata']
labels = {k.split('.')[0]: v['file_attributes']['label'] for k, v in labels.items()}
img_paths = paths.list_images('image_cropped')
imgs = {os.path.basename(p).split('.')[0]: cv2.imread(p) for p in img_paths}
for k in labels.keys():
    test_data.append((imgs[k], labels[k]))

labels = json.load(open('new_images_json.json', 'r'))['_via_img_metadata']
labels = {k.split('.')[0]: v['file_attributes']['label'] for k, v in labels.items()}
img_paths = paths.list_images('new_images')
imgs = {os.path.basename(p).split('.')[0]: cv2.imread(p) for p in img_paths}
for k in labels.keys():
    val_data.append((imgs[k], labels[k]))

labels = {}
for k, v in json.load(open('full_image_json-1.json', 'r'))['_via_img_metadata'].items():
    k = k.split('.')[0]
    for region in v['regions']:
        if len(region['region_attributes']['digit']) == 5:
            label = region['region_attributes']['digit']
            box = region['shape_attributes']
            break
    labels[k] = {
                    'label': label,
                    'box': [box['x'], box['y'], box['width'], box['height']]
                }
img_paths = paths.list_images('full_image')
imgs = {os.path.basename(p).split('.')[0]: cv2.imread(p) for p in img_paths}
for k in labels.keys():
    x, y, w, h = labels[k]['box']
    val_data.append((imgs[k][y:y+h, x:x+w], labels[k]['label']))

labels = {}
with open(r'SCUT-WMN Dataset/easy_samples.txt') as f:
    for line in f.read().strip().split('\n'):
        file_path, label = line.split()
        label = eval(label)
        label = ''.join([str(i) if i < 10 else '?' for i in label])
        labels[os.path.basename(file_path)] = label

for img_path in paths.list_images(r'SCUT-WMN Dataset/easy_samples'):
    # print(img_path)
    img = cv2.imread(img_path)
    train_data.append((img, labels[os.path.basename(img_path)]))

X_test = []
y_test = []
for item in test_data:
    x = resize_to_prefered_height(item[0], img_h)
    label = [int(c) for c in item[1]]
    # y_onehot = enc.transform(label)
    X_test.append(x)
    # y.append(y_onehot[:5])
    y_test.append(label[:5])
    if len(label) == 6: 
        label[-1] = 10
y_test = np.array(y_test)

X_val = []
y_val = []
for item in val_data:
    x = resize_to_prefered_height(item[0], img_h)
    label = [int(c) for c in item[1]]
    # y_onehot = enc.transform(label)
    X_val.append(x)
    # y.append(y_onehot[:5])
    y_val.append(label[:5])
    if len(label) == 6: 
        label[-1] = 10
y_val = np.array(y_val)

X_train = []
y_train = []
for item in train_data:
    x = 255 - resize_to_prefered_height(item[0], img_h)
    label = [10 if c == '?' else int(c) for c in item[1]]
    X_train.append(x)
    y_train.append(label)
y_train = np.array(y_train)


# plt.subplot('311')
# heights = [x[0].shape[0] for x in data]
# print(np.mean(heights))
# plt.hist(heights, bins=len(data))
# plt.subplot('312')
# widths = [x.shape[1] for x in X]
# print(np.mean(widths))
# plt.hist(widths, bins=len(data))
# plt.subplot('313')
# ratios = np.array(widths) / 250
# print(np.mean(ratios))
# plt.hist(ratios, bins=len(data))
# plt.show()
# 
# for x in X:
#     if x.shape[1] / 250 < 2.5 or x.shape[1] / 250 > 5:
#         print(x.shape)
#         plt.imshow(x)
#         plt.show()