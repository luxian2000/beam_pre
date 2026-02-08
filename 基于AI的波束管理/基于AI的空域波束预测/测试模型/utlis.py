import tensorflow as tf
# tf.compat.v1.enable_eager_execution()

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import struct
import operator
import numpy as np
from tensorflow.keras import backend as K
import heapq
import sys
from numpy import random
#Parameter Setting
import os
def seed_tensorflow(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
def mish(x):
    return x * K.tanh(K.softplus(x))
# def top_n(data, n):
#     '''
#     :param rec_list:
#     :param n:
#     :return:
#     '''
#
#     id = [[] for i in tf.range(tf.shape(data)[0])]
#     # rec_list=np.array(rec_list)
#     for i in tf.range(tf.shape(data)[0]):
#         result = heapq.nlargest(n, enumerate(data[i]), operator.itemgetter(1))
#         for j in range(n):
#             id[i].append(result[j][0])
#
#     return id

def top_n(data, n):
    '''
    :param rec_list:
    :param n:
    :return:
    '''
    # id_sort=tf.argsort(data)
    id_sort=np.argsort(data)
    id=id_sort[:,-n:]
    return id

def get_new_data(data, id):
    '''
    获取二维数组中指定id的数据
    :param data:
    :param id:
    :return:
    '''

    # newdata=[[] for i in range(data.shape[1])]
    newdata=np.zeros([512,3])
    for i in range(len(data)):
        for j in range(3):
            # newdata[i].append(data[i][int(id[i][j])])
            p=id[i][j]
            tempdata=data[i][p]
            # newdata[i].append(tf.gather(data,[i][id[i][j]]))
            newdata[i][j]=tempdata
    return tf.convert_to_tensor(newdata)
def mse_id_loss(y_true,y_pred):
    '''

    :param y_true:
    :param y_pred:
    :return:
    '''
    id = tf.argsort(y_true)[:, -3:]
    # id=top_n(y_true,10)
    # y_true_new=get_new_data(y_true,id)
    # y_pred_new=get_new_data(y_pred,id)
    loss=0
    for i in range(512):
        for j in range(3):
            loss+=K.square(y_true[i][id[i][j]]-y_pred[i][id[i][j]])
    res=loss/512
    return res
# def mae_id_loss(y_true,y_pred):
def mse_best_id_loss(y_true,y_pred):
    '''
    :param y_true:
    :param y_pred:
    :return:
    '''
    best_id = tf.reshape(tf.argmax(y_true, 1), [-1, 1])

    line1 = tf.reshape(tf.range(tf.shape(best_id)[0], dtype=tf.int64), [-1, 1])
    y_pred_index = tf.concat((line1, best_id), axis=1)
    y_pred_rsrp = tf.gather_nd(y_pred, y_pred_index)
    y_true_rsrp = tf.reduce_max(y_true, 1)
    y_pred_rsrp = tf.reshape(y_pred_rsrp, [-1, 1])
    y_true_rsrp = tf.reshape(y_true_rsrp, [-1, 1])
    best_id_loss = tf.keras.losses.mean_squared_error(y_true_rsrp, y_pred_rsrp)
    all_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    loss = best_id_loss + all_loss
    return loss

def mse_best_n_id_loss(y_true,y_pred):
    '''

    :param y_true:
    :param y_pred:
    :return:
    '''
    best_id = tf.reshape(tf.argmax(y_true, 1), [-1, 1])
    line1 = tf.reshape(tf.range(tf.shape(best_id)[0], dtype=tf.int32), [-1, 1])
    # line1 = tf.reshape(tf.range(2000, dtype=tf.int64), [-1, 1])
    # top_n_id = tf.convert_to_tensor(top_n(y_true, 5), dtype=tf.int64)
    top_n_id = top_n(y_true, 5)
    y_id1 = tf.reshape(tf.concat((line1, tf.reshape(top_n_id[:, 0], [-1, 1])), axis=1), [-1, 1, 2])
    y_id2 = tf.reshape(tf.concat((line1, tf.reshape(top_n_id[:, 1], [-1, 1])), axis=1), [-1, 1, 2])
    y_id3 = tf.reshape(tf.concat((line1, tf.reshape(top_n_id[:, 2], [-1, 1])), axis=1), [-1, 1, 2])
    y_id4 = tf.reshape(tf.concat((line1, tf.reshape(top_n_id[:, 3], [-1, 1])), axis=1), [-1, 1, 2])
    y_id5 = tf.reshape(tf.concat((line1, tf.reshape(top_n_id[:, 4], [-1, 1])), axis=1), [-1, 1, 2])

    y_id_1 = y_id5
    y_id_2 = tf.concat((y_id4, y_id5), axis=1)
    y_id_3 = tf.concat((y_id3, y_id4, y_id5), axis=1)
    y_id_4 = tf.concat((y_id2,y_id3, y_id4, y_id5), axis=1)
    y_id_5 = tf.concat((y_id1, y_id2, y_id3, y_id4, y_id5), axis=1)

    y_pred_rsrp_1 = tf.gather_nd(y_pred, y_id_1)
    y_pred_rsrp_2 = tf.gather_nd(y_pred, y_id_2)
    y_pred_rsrp_3 = tf.gather_nd(y_pred, y_id_3)
    y_pred_rsrp_4 = tf.gather_nd(y_pred, y_id_4)
    y_pred_rsrp_5 = tf.gather_nd(y_pred, y_id_5)
    y_true_rsrp_1 = tf.gather_nd(y_true, y_id_1)
    y_true_rsrp_2 = tf.gather_nd(y_true, y_id_2)
    y_true_rsrp_3 = tf.gather_nd(y_true, y_id_3)
    y_true_rsrp_4 = tf.gather_nd(y_true, y_id_4)
    y_true_rsrp_5 = tf.gather_nd(y_true, y_id_5)

    id_loss_1 = tf.keras.losses.mean_squared_error(y_true_rsrp_1, y_pred_rsrp_1)
    id_loss_2 = tf.keras.losses.mean_squared_error(y_true_rsrp_2, y_pred_rsrp_2)
    id_loss_3 = tf.keras.losses.mean_squared_error(y_true_rsrp_3, y_pred_rsrp_3)
    id_loss_4 = tf.keras.losses.mean_squared_error(y_true_rsrp_4, y_pred_rsrp_4)
    id_loss_5 = tf.keras.losses.mean_squared_error(y_true_rsrp_5, y_pred_rsrp_5)
    all_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    loss = (id_loss_1+id_loss_2+id_loss_3+id_loss_4+id_loss_5 + all_loss)/6.0
    return loss
def mse_best_n_id_loss2(y_true,y_pred):
    '''

    :param y_true:
    :param y_pred:
    :return:
    '''
    best_id = tf.reshape(tf.argmax(y_true, 1), [-1, 1])
    line1 = tf.reshape(tf.range(tf.shape(best_id)[0], dtype=tf.int32), [-1, 1])
    # line1 = tf.reshape(tf.range(2000, dtype=tf.int64), [-1, 1])
    # top_n_id = tf.convert_to_tensor(top_n(y_true, 5), dtype=tf.int64)
    top_n_id = top_n(y_true, 5)
    y_id1 = tf.reshape(tf.concat((line1, tf.reshape(top_n_id[:, 0], [-1, 1])), axis=1), [-1, 1, 2])
    y_id2 = tf.reshape(tf.concat((line1, tf.reshape(top_n_id[:, 1], [-1, 1])), axis=1), [-1, 1, 2])
    y_id3 = tf.reshape(tf.concat((line1, tf.reshape(top_n_id[:, 2], [-1, 1])), axis=1), [-1, 1, 2])
    y_id4 = tf.reshape(tf.concat((line1, tf.reshape(top_n_id[:, 3], [-1, 1])), axis=1), [-1, 1, 2])
    y_id5 = tf.reshape(tf.concat((line1, tf.reshape(top_n_id[:, 4], [-1, 1])), axis=1), [-1, 1, 2])

    y_id_1 = y_id5
    y_id_2 = y_id4
    y_id_3 = y_id3
    y_id_4 =y_id2
    y_id_5 = tf.concat((y_id1, y_id2, y_id3, y_id4, y_id5), axis=1)

    y_pred_rsrp_1 = tf.gather_nd(y_pred, y_id_1)
    y_pred_rsrp_2 = tf.gather_nd(y_pred, y_id_2)
    y_pred_rsrp_3 = tf.gather_nd(y_pred, y_id_3)
    y_pred_rsrp_4 = tf.gather_nd(y_pred, y_id_4)
    y_pred_rsrp_5 = tf.gather_nd(y_pred, y_id_5)
    y_true_rsrp_1 = tf.gather_nd(y_true, y_id_1)
    y_true_rsrp_2 = tf.gather_nd(y_true, y_id_2)
    y_true_rsrp_3 = tf.gather_nd(y_true, y_id_3)
    y_true_rsrp_4 = tf.gather_nd(y_true, y_id_4)
    y_true_rsrp_5 = tf.gather_nd(y_true, y_id_5)

    id_loss_1 = tf.keras.losses.mean_squared_error(y_true_rsrp_1, y_pred_rsrp_1)
    id_loss_2 = tf.keras.losses.mean_squared_error(y_true_rsrp_2, y_pred_rsrp_2)
    id_loss_3 = tf.keras.losses.mean_squared_error(y_true_rsrp_3, y_pred_rsrp_3)
    id_loss_4 = tf.keras.losses.mean_squared_error(y_true_rsrp_4, y_pred_rsrp_4)
    id_loss_5 = tf.keras.losses.mean_squared_error(y_true_rsrp_5, y_pred_rsrp_5)
    all_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    loss = (id_loss_1+id_loss_2+id_loss_3+id_loss_4+id_loss_5 + all_loss)
    return loss

def mse_best_n_id_loss3(y_true,y_pred):
    '''

    :param y_true:
    :param y_pred:
    :return:
    '''
    best_id = tf.reshape(tf.argmax(y_true, 1), [-1, 1])
    line1 = tf.reshape(tf.range(tf.shape(best_id)[0], dtype=tf.int32), [-1, 1])
    # line1 = tf.reshape(tf.range(2000, dtype=tf.int64), [-1, 1])
    # top_n_id = tf.convert_to_tensor(top_n(y_true, 5), dtype=tf.int64)
    top_n_id = top_n(y_true, 5)
    y_id1 = tf.reshape(tf.concat((line1, tf.reshape(top_n_id[:, 0], [-1, 1])), axis=1), [-1, 1, 2])
    y_id2 = tf.reshape(tf.concat((line1, tf.reshape(top_n_id[:, 1], [-1, 1])), axis=1), [-1, 1, 2])
    y_id3 = tf.reshape(tf.concat((line1, tf.reshape(top_n_id[:, 2], [-1, 1])), axis=1), [-1, 1, 2])
    y_id4 = tf.reshape(tf.concat((line1, tf.reshape(top_n_id[:, 3], [-1, 1])), axis=1), [-1, 1, 2])
    y_id5 = tf.reshape(tf.concat((line1, tf.reshape(top_n_id[:, 4], [-1, 1])), axis=1), [-1, 1, 2])

    y_id_1 = y_id5
    y_id_2 = y_id4
    y_id_3 = y_id3
    y_id_4 =y_id2
    y_id_5 = tf.concat((y_id1, y_id2, y_id3, y_id4, y_id5), axis=1)

    y_pred_rsrp_1 = tf.gather_nd(y_pred, y_id_1)
    y_pred_rsrp_2 = tf.gather_nd(y_pred, y_id_2)
    # y_pred_rsrp_3 = tf.gather_nd(y_pred, y_id_3)
    # y_pred_rsrp_4 = tf.gather_nd(y_pred, y_id_4)
    # y_pred_rsrp_5 = tf.gather_nd(y_pred, y_id_5)
    y_true_rsrp_1 = tf.gather_nd(y_true, y_id_1)
    y_true_rsrp_2 = tf.gather_nd(y_true, y_id_2)
    # y_true_rsrp_3 = tf.gather_nd(y_true, y_id_3)
    # y_true_rsrp_4 = tf.gather_nd(y_true, y_id_4)
    # y_true_rsrp_5 = tf.gather_nd(y_true, y_id_5)

    id_loss_1 = tf.keras.losses.mean_squared_error(y_true_rsrp_1, y_pred_rsrp_1)
    id_loss_2 = tf.keras.losses.mean_squared_error(y_true_rsrp_2, y_pred_rsrp_2)
    all_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    loss = (id_loss_1+id_loss_2 + all_loss)
    return loss

def mse_best_n_id_loss4(y_true,y_pred):
    '''

    :param y_true:
    :param y_pred:
    :return:
    '''
    best_id = tf.reshape(tf.argmax(y_true, 1), [-1, 1])
    line1 = tf.reshape(tf.range(tf.shape(best_id)[0], dtype=tf.int32), [-1, 1])
    top_n_id = top_n(y_true, 5)
    y_id1 = tf.reshape(tf.concat((line1, tf.reshape(top_n_id[:, 0], [-1, 1])), axis=1), [-1, 1, 2])
    y_id2 = tf.reshape(tf.concat((line1, tf.reshape(top_n_id[:, 1], [-1, 1])), axis=1), [-1, 1, 2])
    y_id3 = tf.reshape(tf.concat((line1, tf.reshape(top_n_id[:, 2], [-1, 1])), axis=1), [-1, 1, 2])
    y_id4 = tf.reshape(tf.concat((line1, tf.reshape(top_n_id[:, 3], [-1, 1])), axis=1), [-1, 1, 2])
    y_id5 = tf.reshape(tf.concat((line1, tf.reshape(top_n_id[:, 4], [-1, 1])), axis=1), [-1, 1, 2])

    y_id_1 = y_id5
    y_id_2 = tf.concat((y_id4, y_id5), axis=1)
    y_id_3 = tf.concat((y_id3, y_id4, y_id5), axis=1)
    y_id_4 = tf.concat((y_id2,y_id3, y_id4, y_id5), axis=1)
    y_id_5 = tf.concat((y_id1, y_id2, y_id3, y_id4, y_id5), axis=1)

    y_pred_rsrp_1 = tf.gather_nd(y_pred, y_id_1)
    # y_pred_rsrp_2 = tf.gather_nd(y_pred, y_id_2)
    # y_pred_rsrp_3 = tf.gather_nd(y_pred, y_id_3)
    # y_pred_rsrp_4 = tf.gather_nd(y_pred, y_id_4)
    y_pred_rsrp_5 = tf.gather_nd(y_pred, y_id_5)
    y_true_rsrp_1 = tf.gather_nd(y_true, y_id_1)
    # y_true_rsrp_2 = tf.gather_nd(y_true, y_id_2)
    # y_true_rsrp_3 = tf.gather_nd(y_true, y_id_3)
    # y_true_rsrp_4 = tf.gather_nd(y_true, y_id_4)
    y_true_rsrp_5 = tf.gather_nd(y_true, y_id_5)

    id_loss_1 = tf.keras.losses.mean_squared_error(y_true_rsrp_1, y_pred_rsrp_1)
    # id_loss_2 = tf.keras.losses.mean_squared_error(y_true_rsrp_2, y_pred_rsrp_2)
    # id_loss_3 = tf.keras.losses.mean_squared_error(y_true_rsrp_3, y_pred_rsrp_3)
    # id_loss_4 = tf.keras.losses.mean_squared_error(y_true_rsrp_4, y_pred_rsrp_4)
    id_loss_5 = tf.keras.losses.mean_squared_error(y_true_rsrp_5, y_pred_rsrp_5)
    all_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    loss = (id_loss_1+id_loss_5 + all_loss)
    return loss








class Logger(object):
  def __init__(self, filename="Default.log"):
    self.terminal = sys.stdout
    self.log = open(filename, "a")
  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)
  def flush(self):
    pass
# def BM_1rx_net(x):
#     def add_common_layers(y):
#         y = keras.layers.BatchNormalization()(y)
#         y = keras.layers.LeakyReLU()(y)
#         return y
#     x = keras.layers.Dense(64,activation='relu')(x)
#     x = keras.layers.Dense(128, activation='relu')(x)
#     x = keras.layers.Dense(256, activation='relu')(x)
#     x = keras.layers.Dropout(rate=0.4)(x)
#     x = keras.layers.Dense(64,activation='softmax')(x)
#     return x

# def BM_RSRP_1rx_net(x):
#     def add_common_layers(y):
#         y = keras.layers.BatchNormalization()(y)
#         y = keras.layers.LeakyReLU()(y)
#         return y
#     x = keras.layers.Dense(64,activation='relu')(x)
#     x = keras.layers.Dense(128, activation='relu')(x)
#     x = keras.layers.Dense(256, activation='relu')(x)
#     x = keras.layers.Dropout(rate=0.4)(x)
#     x = keras.layers.Dense(1,activation='linear')(x)
#     return x