import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed, GRU, Dense, Dropout

import convnet
def brutnet(shape=(24, 120, 120, 3), n_out=1):
    
    
    model = tf.keras.Sequential()

    # spatial feature network
    backbone = convnet(shape[1:])
    model.add(TimeDistributed(backbone, input_shape=shape))

    #temporal feature network
    model.add(GRU(64))

    #decision network
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(n_out, activation='sigmoid'))
    model._name = "BrutNet"
    return model