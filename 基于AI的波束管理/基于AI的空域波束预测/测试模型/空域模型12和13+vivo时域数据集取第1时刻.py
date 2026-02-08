from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from RSRP_net import *
from utlis import *
import datetime
import sys
import os
import copy
import h5py

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
p = datetime.datetime.now()
today = p.strftime('%m_%d_%H_%M')
start_time = datetime.datetime.now()

BEAM_RX = 8
BEAM_TX = 32

USE_RX = 6
USE_TX = 1
# rx_idx = [1, 3, 5, 7]#挑选波束为(1,4)
rx_idx = [1, 2, 3, 4, 5, 6]#挑选波束为(1,6)

SMP = 39600
val_split_rate = 0.05
test_split_rate = 0.05

data = h5py.File('data.mat', 'r')
data = np.transpose(data['A'])
data=data.max(1).reshape(39600,1,8)
X = data[:, :, rx_idx]
y_label = np.reshape(data, [-1, 8])

from sklearn.utils import shuffle
TEST_SMP = int(SMP * test_split_rate)
VAL_SMP = int(SMP * val_split_rate)
X, y_label = shuffle(X, y_label, random_state=42)
X = np.reshape(X, [-1, USE_TX, USE_RX])
x_train = X[:SMP - TEST_SMP - VAL_SMP]
x_val = X[SMP - TEST_SMP - VAL_SMP:SMP - TEST_SMP]
x_test = X[SMP - TEST_SMP:]
y_train = y_label[:SMP - TEST_SMP - VAL_SMP]
y_val = y_label[SMP - TEST_SMP - VAL_SMP:SMP - TEST_SMP]
y_test = y_label[SMP - TEST_SMP:]

print("x_train", x_train.shape)
print("x_val", x_val.shape)
print("x_test", x_test.shape)
print("y_train", y_train.shape)
print("y_val", y_val.shape)
print("y_test", y_test.shape)

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


strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    BM_input = keras.Input(shape=(USE_TX, USE_RX, 1))
    BM_RSRP_output = tx_BM_RSRP_resnet4(BM_input)

    BM_RSRP_NET = keras.Model(inputs=(BM_input), outputs=BM_RSRP_output, name='BM_RSRP_NET')
    # BM_RSRP_NET.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss=mse_best_n_id_loss)
    BM_RSRP_NET.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')
    BM_RSRP_NET.summary()

# callbacks
early_stopping = EarlyStopping(monitor='loss', mode='min', verbose=2, patience=40)
reducelronplateau = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=20, verbose=2, mode='auto', cooldown=0,
                                      min_lr=0)
best_model_path = "models/1RSRP-RX" + str(USE_RX) + "TX" + str(USE_TX) + "_" + str(today) + ".h5"
modelcheckpoint = ModelCheckpoint(best_model_path, save_weights_only=True, monitor='val_loss', mode='min', verbose=2,
                                  save_best_only=True)
callbacks_list = [early_stopping, modelcheckpoint, reducelronplateau]

# model training
# hist = BM_RSRP_NET.fit(x=x_train, y=y_train, shuffle=True, batch_size=512, epochs=500, verbose=2,
#                        validation_data=(x_val,y_val),
#                        callbacks=callbacks_list)
# hist.history
# loss = []
# val_loss = []
# loss.append(hist.history['loss'])
# val_loss.append(hist.history['val_loss'])
# np.save('loss/loss' + str(today) + '.npy', loss)
# np.save('loss/val_loss' + str(today) + '.npy', val_loss)

# model test

rsrp_true = y_label[SMP - TEST_SMP:, :]

BM_RSRP_NET.load_weights('models/1RSRP-RX6TX1_10_30_17_49.h5')

# TEST
rsrp_pred = BM_RSRP_NET.predict(x_test)

np.save('RSRP/rsrp_true' + str(today) + '.npy', rsrp_true)
np.save('RSRP/rsrp_pred' + str(today) + '.npy', rsrp_pred)


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
