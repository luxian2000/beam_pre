from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tx_net import *
from utlis import *
import datetime
import sys
import os
# os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2"

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
'''
12*4 base model
'''
p = datetime.datetime.now()
today=p.strftime('%m_%d_%H_%M')
start_time = datetime.datetime.now()
sys.stdout = Logger('Log/'+str(today)+'.txt')
# sys.stdout = Logger('Log/427.txt')
SMP = 39600
# split_rate = 0.05
tx_idx_4_4 = [12,27,32,47]
tx_idx_6_4 = [14, 17,30,33,46,49]

tx_idx_10_4 = [6,9,15,22,25,38,41,48,54,57]


tx_idx_8_4 = [6, 9, 22, 25, 38, 41, 54, 57]
tx_idx_12_4 = [6,9,15, 16,22, 25, 38, 41,47,48, 54, 57]
# tx_idx_12_4 = [6,9,15,22, 25,28,35, 38, 41,48, 54, 57]


data = np.load('data/user/beam/beam_rsrp_all.npy')
data1=data[0:9900,:,0]
data2=data[9900:19800,:,1]
data3=data[19800:29700,:,2]
data4=data[29700:39600,:,3]
data=np.concatenate([data1,data2,data3,data4],0).reshape(39600,64,1)
print('datashape',data.shape)
                    
# data=data[:,:,0].reshape(39600,64,1)
X = data[:, tx_idx_4_4, ::int(BEAM_RX / USE_RX)]
# X = np.reshape(X, [-1, 4, 4])



print(X.shape)
# print(X[0][8][0])
val_split_rate = 0.05
test_split_rate = 0.1
TEST_SMP = int(SMP * test_split_rate)
VAL_SMP = int(SMP * val_split_rate)

y_label = np.reshape(data, [-1, 64])
X,y_label=shuffle(X,y_label,random_state=42)

x_train = X[:SMP-TEST_SMP]
# x_train = X[:10000]
# x_val=X[SMP-TEST_SMP-VAL_SMP:SMP-TEST_SMP]
# x_test=X[SMP-TEST_SMP:]
x_test=X[SMP-TEST_SMP:]
# y_label = np.load('beam_best_rsrp.npy')
# y_label = np.transpose(y_label['rsrp_best'][:])

y_train = y_label[:SMP-TEST_SMP]
# y_train = y_label[:10000]
# y_val=y_label[SMP-TEST_SMP-VAL_SMP:SMP-TEST_SMP]
y_test=y_label[SMP-TEST_SMP:]

print(x_train.shape)
print(y_label.shape)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    BM_input = keras.Input(shape=(4, 1,1))
    BM_RSRP_output = BM_RSRP_resnet4(BM_input)
    # BM_RSRP_output = model_compare(BM_input)
    BM_RSRP_NET = keras.Model(inputs=(BM_input), outputs=BM_RSRP_output, name='BM_RSRP_NET')
    # BM_RSRP_NET.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss=mse_best_n_id_loss3)
    # BM_RSRP_NET.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mae')
    BM_RSRP_NET.summary()
# print('using mse_best_n_id_loss2')
# callbacks
# BM_RSRP_NET.load_weights('models/1RSRP-TX12RX4_05_21_21_37.h5')
# for layer in BM_RSRP_NET.layers[:8]:
#     layer.trainable = True
# for layer in BM_RSRP_NET.layers[8:]:
#     layer.trainable = False
BM_RSRP_NET.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss=mse_best_n_id_loss3)

early_stopping = EarlyStopping(monitor='loss', mode='min', verbose=2, patience=45)
reducelronplateau = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=20, verbose=2, mode='auto', cooldown=0,
                                      min_lr=0)
best_model_path =  "models/tx_beam_4_op2" +  "_" + str(today) + ".h5"
modelcheckpoint = ModelCheckpoint(best_model_path, save_weights_only=False, monitor='val_loss', mode='min', verbose=2,
                                  save_best_only=True)
callbacks_list = [early_stopping, modelcheckpoint, reducelronplateau]

# model training
hist=BM_RSRP_NET.fit(x=x_train, y=y_train,validation_data=(x_test,y_test),shuffle=True, batch_size=512, epochs=500, verbose=2,callbacks=callbacks_list)
hist.history
loss=[]
val_loss=[]
loss.append(hist.history['loss'])
val_loss.append(hist.history['val_loss'])
np.save('loss/loss_4_op2'+'.npy',loss)
np.save('loss/val_loss_4_op2'+'.npy',val_loss)
# model test
TEST_SMP = int(SMP * split_rate)
# x_test = x_train[SMP - TEST_SMP:, :]
# rsrp_true = y_label[SMP - TEST_SMP:, :]

BM_RSRP_NET.load_weights(best_model_path)

# TEST
rsrp_pred = BM_RSRP_NET.predict(x_test)
rsrp_err = abs(rsrp_pred - y_test)
print('全部RSRP预测误差 = ', np.mean(rsrp_err))

np.save('RSRP/y_test_4_op2'+'.npy', y_test)
np.save('RSRP/rsrp_pred_4_op2'+'.npy', rsrp_pred)
# 全部误差，1980*256
rsrp_err = abs(rsrp_pred - y_test)
# 单个id误差
idx_error = np.mean(rsrp_err, axis=0)
# 单个数据误差，1980，
sample_error = np.mean(rsrp_err, axis=1)

# 最优RSRP误差
rsrp_err = abs(y_test.max(axis=1) - rsrp_pred.max(axis=1))
print('最优RSRP预测误差 = ', np.mean(rsrp_err))
rsrp_best_id = y_test.argmax(1)
rsrp_pred_id = rsrp_pred.argmax(1)
right_error=0
num1=0
num2=0
wrong_error=0
for i in range(len(rsrp_best_id)):
    if rsrp_best_id[i]==rsrp_pred_id[i]:
        right_error+=abs((y_test[i][rsrp_best_id[i]]-rsrp_pred[i][rsrp_pred_id[i]]))
        num1+=1
    else:
        wrong_error+=abs((y_test[i][rsrp_best_id[i]]-rsrp_pred[i][rsrp_pred_id[i]]))
        num2+=1
print('预测正确误差: ',right_error/num1)
print('预测错误误差: ',wrong_error/num2)


N = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

num = []
def test_top_n(data, n):
    '''
    :param rec_list:
    :param n:
    :return:
    '''
    id_sort=np.argsort(data)
    id=id_sort[:,-n:]
    return id
for i in N:
    id_n = test_top_n(rsrp_pred, i)
    ans = 0
    for j in range(TEST_SMP):
        if rsrp_best_id[j] in id_n[j]:
            ans += 1
    num.append(ans)
for i in range(len(num)):
    print(num[i] / TEST_SMP)

end_time = datetime.datetime.now()
total_time = end_time - start_time
# print('using mse_best_n_id_loss3 and 6,4')
print(end_time)
print('time used: ', total_time)
