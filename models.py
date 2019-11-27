from keras.models import Sequential, Model
from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization,
                          Flatten, GlobalMaxPool2D, MaxPool2D, concatenate,
                          Activation, Input, Dense, Dropout, TimeDistributed,
                          Bidirectional, LSTM, GlobalAveragePooling2D, GRU,
                          Convolution1D, MaxPool1D, GlobalMaxPool1D, MaxPooling2D,
                          Reshape, Lambda)
from keras import optimizers
from keras.utils import Sequence, to_categorical
from keras.regularizers import l2
from keras.initializers import random_normal
from keras.activations import relu
from keras import backend as K

# Define CTC loss
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def ctc(y_true, y_pred):
    return y_pred

def model7(img_h, training=True):
    from inception_blocks_model7 import conv2d_bn, block_a, block_b
    inp = Input(shape=(None, img_h, 3), name='input')

    x = block_a(inp, 16, pool_size=(2,2))
    x = block_b(x, 16, pool_size=(2,2))
    x = block_b(x, 32, pool_size=(2,2))
    x = block_b(x, 64, pool_size=(2,2))
    x = block_b(x, 128, pool_size=(2,2))
    x = block_b(x, 128, pool_size=(2,2))
    # x = conv2d_bn(x, 128, (5,11))
    # x = MaxPool2D((1,2))(x)

    # frame = inspect.currentframe()
    # model_name = inspect.getframeinfo(frame).function
    x = Reshape((-1, img_h // 2**6 * 128 * 3))(x)

    x = TimeDistributed(Dense(256, activation='elu'))(x)
    # x = LSTM(256, return_sequences=True, activation='tanh')(x)
    x = GRU(256, return_sequences=True, activation='tanh')(x)
    x = Dropout(0.5)(x)
    # x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'), merge_mode='concat')(x)
    # x = Dropout(0.2)(x)
    # x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'), merge_mode='concat')(x)
    # x = Dropout(0.5)(x)
    # x = TimeDistributed(Dense(256, activation='relu'))(x)
    # x = Dropout(0.2)(x)
    pred = TimeDistributed(Dense(12, activation='softmax'))(x)

    labels = Input(shape=(None,), dtype='int32', name='labels')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([pred,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])

    if training:
        model = Model([inp, labels, input_length, label_length], loss_out)
        opt = optimizers.Nadam(0.001)
        model.compile(optimizer=opt, loss=ctc)
    else:
        model = Model(inp, pred)

    return model


def model0(img_h, training=True):
    inp = Input(shape=(None, img_h, 3), name='input')

    x = Convolution2D(32, (3,3), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D((2,2))(x)

    x = Convolution2D(64, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D((2,2))(x)

    x = Convolution2D(128, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D((2,2))(x)

    # x = Convolution2D(64, (3,3), padding="same")(x)
    # x = BatchNormalization()(x)
    # x = Activation("relu")(x)
    # x = MaxPool2D((2,2))(x)
    # 
    # x = Convolution2D(128, (3,3), padding="same")(x)
    # x = BatchNormalization()(x)
    # x = Activation("relu")(x)
    # x = MaxPool2D((2,2))(x)
    # 
    # x = Convolution2D(128, (3,3), padding="same")(x)
    # x = BatchNormalization()(x)
    # x = Activation("relu")(x)
    # x = MaxPool2D((2,2))(x)

    x = Reshape((-1, img_h // 2**3 * 128))(x)

    x = LSTM(512, return_sequences=True, activation='tanh')(x)
    pred = TimeDistributed(Dense(12, activation='softmax'))(x)

    labels = Input(shape=(None,), dtype='int32', name='labels')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([pred,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])

    if training:
        model = Model([inp, labels, input_length, label_length], loss_out)
        # opt = optimizers.Nadam(0.01)
        model.compile(optimizer='adam', loss=ctc)
    else:
        model = Model(inp, pred)

    return model

def get_classifier_model(input_size=32, num_classes=2, num_filters=32):
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
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    
    return model

if __name__ == '__main__':
    model = get_classifier_model(num_classes=2)
    print(model.summary())