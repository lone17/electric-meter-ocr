import cv2
import math
import random
import pickle
import keras
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

def pad(img, timesteps, value):
    img = cv2.copyMakeBorder(img, 0, 0, 0, timesteps - img.shape[1],
                              cv2.BORDER_CONSTANT, value=value)

    return img

class DataGenerator(keras.utils.Sequence):

    def __init__(self, X, y, down_width_factor, batch_size=32, val_size=0.2,
                 preprocess=lambda x: x, phase='test'):
        self.preprocess = preprocess
        self.X = np.empty(len(X), dtype=object)
        for i, x in enumerate(X):
            self.X[i] = x
        self.y = np.array(y)
        self.batch_size = batch_size
        self.down_width_factor = down_width_factor
        self.shuffle_each_epoch = (phase == 'train')
        if self.shuffle_each_epoch:
            self.indices = np.random.permutation(len(X))
        else:
            self.indices = list(range(len(X)))
        self.label_length = np.array([len(label) for label in y])
        
        
        # self.pad = False
        # # X = np.array(X)
        # # y = np.array(y)
        # y_length = np.array([len(label) for label in y])
        # permu = list(range(len(y)))
        # random.shuffle(permu)
        # 
        # # self.X_train, self.X_val, self.y_train, self.y_val = \
        # # train_test_split(X, y, test_size=val_size, shuffle=True)
        # split = int(len(permu) * val_size)
        # train_idx = permu[:-split]
        # val_idx = permu[-split:]
        # 
        # if self.pad:
        #     self.X_train = X[train_idx]
        #     self.X_val = X[val_idx]
        # else:
        #     self.X_train = [X[i] for i in train_idx]
        #     self.X_val = [X[i] for i in val_idx]
        # self.y_train = y[train_idx]
        # self.y_val = y[val_idx]
        # self.y_train_length = y_length[train_idx]
        # self.y_val_length = y_length[val_idx]
        # 
        # # self.input_length = X.shape[1] // down_width_factor - 2
        # del X
        # del y
        # 
        # self.batch_size = batch_size
        # self.train_size = len(self.X_train)
        # self.val_size = len(self.X_val)
        # self.train_steps = math.ceil(self.train_size / batch_size)
        # self.val_steps = math.ceil(self.val_size / batch_size)
        
    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        
        cur_indices = self.indices[start_idx: end_idx]
        X = self.X[cur_indices]
        y = self.y[cur_indices]
        label_length = self.label_length[cur_indices]
        
        batch_max_timestep = max([img.shape[1] for img in X])
        X = np.array([pad(img, batch_max_timestep, 0) for img in X], dtype='float')
        X = self.preprocess(X)
        X = np.transpose(X, (0, 2, 1, 3))
        
        input_length = batch_max_timestep // self.down_width_factor - 2
        input_length = np.ones([len(cur_indices), 1]) * input_length

        inputs = {
            'input': X,
            'labels': y,
            'input_length': input_length,
            'label_length': label_length,
            }

        outputs = {'ctc': np.zeros([len(cur_indices)])}

        return inputs, outputs
    
    def on_epoch_end(self):
        if self.shuffle_each_epoch:
            random.shuffle(self.indices)

    def next_train(self):
        while True:
            for i in range(0, self.train_size, self.batch_size):
                X = self.X_train[i : i+self.batch_size]
                y = self.y_train[i : i+self.batch_size]
                label_length = self.y_train_length[i : i+self.batch_size]

                batch_size = len(X)
                if self.pad:
                    batch_max_timestep = X.shape[0]
                else:
                    batch_max_timestep = max([img.shape[1] for img in X])
                    X = np.array([pad(img, batch_max_timestep, True) for img in X], dtype='uint8')
                    X = np.transpose(X, (0, 2, 1, 3))

                input_length = batch_max_timestep // self.down_width_factor - 2
                input_length = np.ones([batch_size, 1]) * input_length

                inputs = {
                    'input': X,
                    'labels': y,
                    'input_length': input_length,
                    'label_length': label_length,
                    }

                outputs = {'ctc': np.zeros([batch_size])}

                yield (inputs, outputs)

    def next_val(self):
        while True:
            for i in range(0, self.val_size, self.batch_size):
                X = self.X_val[i : i+self.batch_size]
                y = self.y_val[i : i+self.batch_size]
                label_length = self.y_val_length[i : i+self.batch_size]

                batch_size = len(X)
                if self.pad:
                    batch_max_timestep = X.shape[1]
                else:
                    batch_max_timestep = max([img.shape[1] for img in X])
                    X = np.array([pad(img, batch_max_timestep, True) for img in X], dtype='uint8')
                    X = np.transpose(X, (0, 2, 1, 3))

                input_length = batch_max_timestep // self.down_width_factor - 2
                input_length = np.ones([batch_size, 1]) * input_length

                inputs = {
                    'input': X,
                    'labels': y,
                    'input_length': input_length,
                    'label_length': label_length,
                    }

                outputs = {'ctc': np.zeros([batch_size])}

                yield (inputs, outputs)