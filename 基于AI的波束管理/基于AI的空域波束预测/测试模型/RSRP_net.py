import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import struct
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects
#Parameter Setting
# BEAM_RX = 4
# BEAM_TX = 64

# USE_RX = 4
# USE_TX = 8
# tx_idx = [6,9,22,25,38,41,54,57]
# SMP = 39600
# split_rate = 0.05


def mish(x):
    return x * K.tanh(K.softplus(x))

class Mish(Activation):
    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


get_custom_objects().update({'Mish': Mish(mish)})

def BM_RSRP_net(x):
    def add_common_layers(y):
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.LeakyReLU()(y)
        return y

    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Flatten()(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    #x = keras.layers.Dropout(rate=0.4)(x)
    x = keras.layers.Dense(256,activation='linear')(x)
    return x



def BM_RSRP_resnet(x):

    x = layers.Conv2D(64, kernel_size=(1, 1), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    shortcut = x
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(64, kernel_size=(5, 5), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.25)(x)

    shortcut = x
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(64, kernel_size=(5, 5), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.25)(x)

    shortcut = x
    shortcut = layers.Conv2D(128, kernel_size=(1, 1), padding='same', data_format='channels_last')(shortcut)
    shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same',  data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, kernel_size=(5, 5), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.25)(x)

    shortcut = x
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same', strides=1, data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, kernel_size=(5, 5), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.25)(x)

    shortcut = x
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same', strides=1, data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, kernel_size=(5, 5), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.25)(x)

    shortcut = x
    shortcut = layers.Conv2D(256, kernel_size=1, strides=2, use_bias=False)(shortcut)
    shortcut = layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5)(shortcut)
    x = layers.Conv2D(256, kernel_size=(3, 3), padding='same', strides=2, data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.25)(x)

    shortcut = x
    x = layers.Conv2D(256, kernel_size=(3, 3), padding='same', strides=1, data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.25)(x)

    # x = layers.GlobalAveragePooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='linear')(x)
    return x
def BM_RSRP_resnet2(x):

    x = layers.Conv2D(64, kernel_size=(1, 1), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    shortcut = x
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.25)(x)

    shortcut = x
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.25)(x)

    shortcut = x
    shortcut = layers.Conv2D(128, kernel_size=(1, 1), padding='same', data_format='channels_last')(shortcut)
    shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same',  data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.25)(x)

    shortcut = x
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same', strides=1, data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, kernel_size=(5, 5), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.25)(x)

    shortcut = x
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same', strides=1, data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.25)(x)

    shortcut = x
    shortcut = layers.Conv2D(256, kernel_size=1, strides=2, use_bias=False)(shortcut)
    shortcut = layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5)(shortcut)
    x = layers.Conv2D(256, kernel_size=(3, 3), padding='same', strides=2, data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.25)(x)

    shortcut = x
    x = layers.Conv2D(256, kernel_size=(3, 3), padding='same', strides=1, data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.25)(x)

    # x = layers.GlobalAveragePooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='linear')(x)
    return x
def BM_RSRP_resnet3(x):

    x = layers.Conv2D(64, kernel_size=(1, 1), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    shortcut = x
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.25)(x)

    shortcut = x
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.25)(x)

    shortcut = x
    shortcut = layers.Conv2D(128, kernel_size=(1, 1), padding='same', data_format='channels_last')(shortcut)
    shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same',  data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.25)(x)

    shortcut = x
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same', strides=1, data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.25)(x)

    shortcut = x
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same', strides=1, data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.25)(x)

    shortcut = x
    shortcut = layers.Conv2D(256, kernel_size=1, strides=2, use_bias=False)(shortcut)
    shortcut = layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5)(shortcut)
    x = layers.Conv2D(256, kernel_size=(3, 3), padding='same', strides=2, data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.25)(x)

    shortcut = x
    x = layers.Conv2D(256, kernel_size=(3, 3), padding='same', strides=1, data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.25)(x)

    # x = layers.GlobalAveragePooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='linear')(x)
    return x
def add_common_layers(x):
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    return x
def BM_RSRP_resnet4(x):

    x = layers.Conv2D(64, kernel_size=(1, 1), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    shortcut = x
    for i in range(4):
        x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
        x=add_common_layers(x)
    x = layers.Add()([shortcut, x])
    x = layers.Dropout(rate=0.25)(x)
    shortcut=x
    shortcut = layers.Conv2D(128, kernel_size=(1, 1), padding='same', data_format='channels_last')(shortcut)
    for i in range(4):
        x = layers.Conv2D(128, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
        x = add_common_layers(x)
    x = layers.Add()([shortcut, x])
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same', strides=2, data_format='channels_last')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='linear')(x)
    return x

def tx_BM_RSRP_resnet4(x):

    x = layers.Conv2D(64, kernel_size=(1, 1), padding='same', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    shortcut = x
    for i in range(4):
        x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
        x=add_common_layers(x)
    x = layers.Add()([shortcut, x])
    x = layers.Dropout(rate=0.25)(x)
    shortcut=x
    shortcut = layers.Conv2D(128, kernel_size=(1, 1), padding='same', data_format='channels_last')(shortcut)
    for i in range(4):
        x = layers.Conv2D(128, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
        x = add_common_layers(x)
    x = layers.Add()([shortcut, x])
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same', strides=2, data_format='channels_last')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(8, activation='linear')(x)
    return x

if __name__ == '__main__':
    Pos_input = keras.Input(shape=(8, 4, 1))
    # Pos_output = BM_RSRP_resnet4(Pos_input)
    Pos_output = model_compare(Pos_input)
    model = keras.Model(inputs=Pos_input, outputs=Pos_output, name='Pos_net')
    model.summary()
    from keras_flops import get_flops
    flops=get_flops(model,batch_size=1)
    print(flops)
