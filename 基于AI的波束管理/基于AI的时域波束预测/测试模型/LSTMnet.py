import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import struct
import numpy as np
from tensorflow.keras import backend as K


def mish(x):
    return x * K.tanh(K.softplus(x))


def add_common_layers(x):
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    return x


def ConvLSTMnet(x):
    x = layers.ConvLSTM2D(filters=32,kernel_size=(3,3),strides=2,padding='same',return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=2,padding='same',return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(filters=64,kernel_size=(3, 3), strides=2, padding='same', return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    # x = layers.GlobalMaxPooling3D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(2048, activation='linear')(x)
    return x
def LSTM_net(x):
    x = layers.LSTM(units=128,return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(units=256, return_sequences=True)(x)
    # x = layers.LSTM(units=512, return_sequences=False)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(128, activation='linear')(x)
    return x
def LSTM_F4_net(x):
    x=layers.ConvLSTM2D(filters=30,kernel_size=(3,3),padding='same',return_sequences=True)(x)
    x=layers.BatchNormalization()(x)

    x = layers.ConvLSTM2D(filters=30, kernel_size=(3, 3), padding='same',return_sequences=True)(x)
    x = layers.BatchNormalization()(x)

    x = layers.ConvLSTM2D(filters=30, kernel_size=(3, 3), padding='same', return_sequences=False)(x)
    x= layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(256*4, activation='linear')(x)
    return x

def Tx_LSTM_F4_net(x):
    x=layers.ConvLSTM2D(filters=30,kernel_size=(3,3),padding='same',return_sequences=True)(x)
    x=layers.BatchNormalization()(x)

    x = layers.ConvLSTM2D(filters=30, kernel_size=(3, 3), padding='same',return_sequences=True)(x)
    x = layers.BatchNormalization()(x)

    x = layers.ConvLSTM2D(filters=30, kernel_size=(3, 3), padding='same', return_sequences=False)(x)
    x= layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(32*4, activation='linear')(x)
    return x
def LSTM_F2_net(x):
    x=layers.ConvLSTM2D(filters=30,kernel_size=(3,3),padding='same',return_sequences=True)(x)
    x=layers.BatchNormalization()(x)

    x = layers.ConvLSTM2D(filters=30, kernel_size=(3, 3), padding='same',return_sequences=True)(x)
    x = layers.BatchNormalization()(x)

    x = layers.ConvLSTM2D(filters=30, kernel_size=(3, 3), padding='same', return_sequences=False)(x)
    x= layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(256*2, activation='linear')(x)
    return x
def Tx_LSTM_F2_net(x):
    x=layers.ConvLSTM2D(filters=30,kernel_size=(3,3),padding='same',return_sequences=True)(x)
    x=layers.BatchNormalization()(x)

    x = layers.ConvLSTM2D(filters=30, kernel_size=(3, 3), padding='same',return_sequences=True)(x)
    x = layers.BatchNormalization()(x)

    x = layers.ConvLSTM2D(filters=30, kernel_size=(3, 3), padding='same', return_sequences=False)(x)
    x= layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(32*2, activation='linear')(x)
    return x
def LSTM_F1_net(x):
    x=layers.ConvLSTM2D(filters=30,kernel_size=(3,3),padding='same',return_sequences=True)(x)
    x=layers.BatchNormalization()(x)

    x = layers.ConvLSTM2D(filters=30, kernel_size=(3, 3), padding='same',return_sequences=True)(x)
    x = layers.BatchNormalization()(x)

    x = layers.ConvLSTM2D(filters=30, kernel_size=(3, 3), padding='same', return_sequences=False)(x)
    x= layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(256, activation='linear')(x)
    return x
def Tx_LSTM_F1_net(x):
    x=layers.ConvLSTM2D(filters=30,kernel_size=(3,3),padding='same',return_sequences=True)(x)
    x=layers.BatchNormalization()(x)

    x = layers.ConvLSTM2D(filters=30, kernel_size=(3, 3), padding='same',return_sequences=True)(x)
    x = layers.BatchNormalization()(x)

    x = layers.ConvLSTM2D(filters=30, kernel_size=(3, 3), padding='same', return_sequences=False)(x)
    x= layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(32*1, activation='linear')(x)
    return x
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
def stats_graph(graph):
    opt1 = tf.profiler.ProfileOptionBuilder.float_operation()
    opt2 = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
    flops=tf.profiler.profile(graph=None,options=opt1)
    params= tf.profiler.profile(graph=None,options=opt2)
    # print('GFLOPs: {};  Trainable params: {}'.format(flops.total_float_ops/1000000000.0,params.total_parameters))
    return flops.total_float_ops,params.total_parameters
if __name__ == '__main__':
    # BM_input = keras.Input(shape=(8,32,8,1))
    # BM_output=ConvLSTMnet(BM_input)
    BM_input = keras.Input(shape=(4,8,4,1))
    # BM_output=LSTM_F1_net(BM_input)
    BM_output=LSTM_F1_net(BM_input)
    model = keras.Model(inputs=BM_input, outputs=BM_output, name='BM_net')
    model.summary()
    print('计算量：', stats_graph(model))
    print('计算量2:', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

