from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from RSRP_net import *
from utlis import *
import datetime
import sys
import os

# os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
p = datetime.datetime.now()
today = p.strftime('%m_%d_%H_%M')
start_time = datetime.datetime.now()
sys.stdout = Logger('Log/' + str(today) + '.txt')
# sys.stdout = Logger('Log/427.txt')
BEAM_RX = 4
BEAM_TX = 64

USE_RX = 4
USE_TX = 8
tx_idx = [6, 9, 22, 25, 38, 41, 54, 57]
tx_idx_2=[0,9,18,27,36,45,54,63]
tx_idx_3=[0,1,2,3,60,61,62,63]
tx_idx_4=[20,21,22,23,40,41,42,43]
tx_idx_5=[0,3,20,23,40,43,60,63]

SMP = 39600*5
val_split_rate = 0.05
test_split_rate = 0.05
data = np.load('beam_rsrp_all.npy')
from sklearn.utils import shuffle

TEST_SMP = int(SMP * test_split_rate)
VAL_SMP = int(SMP * val_split_rate)
X1 = data[:, tx_idx, ::int(BEAM_RX / USE_RX)]
X2 = data[:, tx_idx_2, ::int(BEAM_RX / USE_RX)]
X3 = data[:, tx_idx_3, ::int(BEAM_RX / USE_RX)]
X4 = data[:, tx_idx_4, ::int(BEAM_RX / USE_RX)]
X5 = data[:, tx_idx_5, ::int(BEAM_RX / USE_RX)]
y_label = np.reshape(data, [-1, 256])
X=np.concatenate((X1,X2,X3,X4,X5),0)
y_label=np.concatenate((y_label,y_label,y_label,y_label,y_label),0)
X, y_label = shuffle(X, y_label, random_state=42)
X = np.reshape(X, [-1, USE_TX, USE_RX])
x_train = X[:SMP - TEST_SMP - VAL_SMP]
x_val = X[SMP - TEST_SMP - VAL_SMP:SMP - TEST_SMP]
x_test = X[SMP - TEST_SMP:]
y_train = y_label[:SMP - TEST_SMP - VAL_SMP]
y_val = y_label[SMP - TEST_SMP - VAL_SMP:SMP - TEST_SMP]
y_test = y_label[SMP - TEST_SMP:]

print(x_train.shape)
print(y_label.shape)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    BM_input = keras.Input(shape=(USE_TX, USE_RX, 1))
    BM_RSRP_output = BM_RSRP_resnet4(BM_input)
    BM_RSRP_NET = keras.Model(inputs=(BM_input), outputs=BM_RSRP_output, name='BM_RSRP_NET')
    BM_RSRP_NET.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss=mse_best_n_id_loss)
    # BM_RSRP_NET.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mae')
    BM_RSRP_NET.summary()

# callbacks
early_stopping = EarlyStopping(monitor='loss', mode='min', verbose=2, patience=40)
reducelronplateau = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=20, verbose=2, mode='auto', cooldown=0,
                                      min_lr=0)
best_model_path = "models/1RSRP-TX" + str(USE_TX) + "RX" + str(USE_RX) + "_" + str(today) + ".h5"
modelcheckpoint = ModelCheckpoint(best_model_path, save_weights_only=True, monitor='val_loss', mode='min', verbose=2,
                                  save_best_only=True)
callbacks_list = [early_stopping, modelcheckpoint, reducelronplateau]

# model training
hist = BM_RSRP_NET.fit(x=x_train, y=y_label, shuffle=True, batch_size=512, epochs=500, verbose=2,
                       validation_split=split_rate,
                       callbacks=callbacks_list)
hist.history
loss = []
val_loss = []
loss.append(hist.history['loss'])
val_loss.append(hist.history['val_loss'])
np.save('loss/loss' + str(today) + '.npy', loss)
np.save('loss/val_loss' + str(today) + '.npy', val_loss)
# model test
# x_test = x_train[SMP - TEST_SMP:, :]
rsrp_true = y_label[SMP - TEST_SMP:, :]

BM_RSRP_NET.load_weights(best_model_path)

# TEST
rsrp_pred = BM_RSRP_NET.predict(x_test)
rsrp_err = abs(rsrp_pred - rsrp_true)
print('全部RSRP预测误差 = ', np.mean(rsrp_err))

np.save('RSRP/rsrp_true' + str(today) + '.npy', rsrp_true)
np.save('RSRP/rsrp_pred' + str(today) + '.npy', rsrp_pred)
# 全部误差，1980*256
rsrp_err = abs(rsrp_pred - rsrp_true)
# 单个id误差
idx_error = np.mean(rsrp_err, axis=0)
# 单个数据误差，1980，
sample_error = np.mean(rsrp_err, axis=1)

# 最优RSRP误差
rsrp_err = abs(rsrp_true.max(axis=1) - rsrp_pred.max(axis=1))
print('RSRP预测误差 = ', np.mean(rsrp_err))

N = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rsrp_best_id = rsrp_true.argmax(1)
num = []
for i in N:
    id_n = top_n(rsrp_pred, i)
    ans = 0
    for j in range(TEST_SMP):
        if rsrp_best_id[j] in id_n[j]:
            ans += 1
    num.append(ans)
for i in range(len(num)):
    print('前' + str(i + 1) + '个预测结果中选择正确波速对的数目为：' + str(num[i]) + ' 概率为: ', num[i] / TEST_SMP)

end_time = datetime.datetime.now()
total_time = end_time - start_time
print(end_time)
print('time used: ', total_time)
