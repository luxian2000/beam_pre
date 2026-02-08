from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from RSRP_net import *
from LSTMnet import *
from utlis import *
import datetime
import sys
import os
import copy
from sklearn.utils import shuffle
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

p = datetime.datetime.now()
today = p.strftime('%m_%d_%H_%M')
start_time = datetime.datetime.now()
sys.stdout = Logger('Log/' + str(today) + '.txt')
sys.stdout = Logger('Log/427.txt')

tx_idx_8 = [2, 5, 10, 13, 18, 21, 26, 29]

val_split_rate = 0.05
test_split_rate = 0.05

data4_all = np.load('data_30km/beam_rsrp_all_30.npy') #(1050, 100, 32, 4)
# data4_all=data4_all.max(3).reshape(1050, 100, 32, 1)
data4_id = np.load('data_30km/beam_best_idx_30.npy')

for i in range(25-1):
    if i==0:
        x=data4_all[:,4*i:4*i+4,:,:]
        y_f1=data4_all[:,4*(i+1),:,:]
        y_f2=data4_all[:,4*(i+1)+1,:,:]
        y_f3=data4_all[:,4*(i+1)+2,:,:]
        y_f4=data4_all[:,4*(i+1)+3,:,:]
        #y_all=np.concatenate((y_f1,y_f2,y_f3,y_f4),axis=1)
    else:
        x=np.concatenate((x,data4_all[:,4*i:4*i+4,:,:]),axis=0)
        y_f1=np.concatenate((y_f1,data4_all[:,4*(i+1),:,:]),axis=0)
        y_f2=np.concatenate((y_f2,data4_all[:,4*(i+1)+1,:,:]),axis=0)
        y_f3=np.concatenate((y_f3,data4_all[:,4*(i+1)+2,:,:]),axis=0)
        y_f4=np.concatenate((y_f4,data4_all[:,4*(i+1)+3,:,:]),axis=0)
        y_all=np.concatenate((y_f1,y_f2,y_f3,y_f4),axis=1) # (25200, 32+32+32+32, 4)
        #y_all=np.concatenate((y_all,y_all_temp),axis=0)

x=x.max(3).reshape(25200, 4, 32, 1, 1)#接收波束取最优

#y=y_f4.reshape(25200,128)
y_all=y_all.max(2)
y=y_all.reshape(25200,32*4)
x, y = shuffle(x, y, random_state=42)
x = x[:,:, tx_idx_8,:]
print('x:',x.shape)
print('y:',y.shape)
print('y_all:',y_all.shape)
print('y_f1:',y_f1.shape)
SMP=25200

TEST_SMP = int(SMP * test_split_rate)
VAL_SMP = int(SMP * val_split_rate)

x_train = x[:SMP - TEST_SMP - VAL_SMP]
x_val = x[SMP - TEST_SMP - VAL_SMP:SMP - TEST_SMP]
x_test = x[SMP - TEST_SMP:]
# y_label = np.load('beam_best_rsrp.npy')
# y_label = np.transpose(y_label['rsrp_best'][:])

y_train = y[:SMP - TEST_SMP - VAL_SMP]
y_val = y[SMP - TEST_SMP - VAL_SMP:SMP - TEST_SMP]
y_test = y[SMP - TEST_SMP:]

print(x_train.shape)#(22680, 4, 8, 1, 1)
print(y_train.shape)#(22680, 128)
def LSTM_F4__net(x):
    x=layers.ConvLSTM2D(filters=30,kernel_size=(3,3),padding='same',return_sequences=True)(x)
    x=layers.BatchNormalization()(x)

    x = layers.ConvLSTM2D(filters=30, kernel_size=(3, 3), padding='same',return_sequences=True)(x)
    x = layers.BatchNormalization()(x)

    x = layers.ConvLSTM2D(filters=30, kernel_size=(3, 3), padding='same', return_sequences=False)(x)
    x= layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(32*4, activation='linear')(x)
    return x


seed_tensorflow(42)
# seed_tensorflow(50)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    BM_input = keras.Input(shape=(4,8,1,1))
    BM_RSRP_output = LSTM_F4__net(BM_input)
    BM_RSRP_NET = keras.Model(inputs=(BM_input), outputs=BM_RSRP_output, name='BM_RSRP_NET')
    # BM_RSRP_NET.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss=mse_best_n_id_loss)
    BM_RSRP_NET.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')
    BM_RSRP_NET.summary()

# callbacks
early_stopping = EarlyStopping(monitor='loss', mode='min', verbose=2, patience=45)
reducelronplateau = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=20, verbose=2, mode='auto', cooldown=0,
                                      min_lr=0)
best_model_path = "models/30Tx8Rx1"+ "_" + str(today) + ".h5"
modelcheckpoint = ModelCheckpoint(best_model_path, save_weights_only=True, monitor='val_loss', mode='min', verbose=2,
                                  save_best_only=True)
callbacks_list = [early_stopping, modelcheckpoint, reducelronplateau]

# model training
hist = BM_RSRP_NET.fit(x=x_train, y=y_train, shuffle=True, batch_size=126, epochs=500, verbose=2,validation_data=(x_val,y_val),
                       callbacks=callbacks_list)
hist.history
loss = []
val_loss = []
loss.append(hist.history['loss'])
val_loss.append(hist.history['val_loss'])
np.save('loss/loss' + str(today) + '.npy', loss)
np.save('loss/val_loss' + str(today) + '.npy', val_loss)
rsrp_true = y_test
rsrp_pred = BM_RSRP_NET.predict(x_test, batch_size=126)
rsrp_err = abs(rsrp_pred - rsrp_true)
print('全部RSRP预测误差 = ', np.mean(rsrp_err))

np.save('RSRP/30_rsrp_true' + str(today) + '.npy', rsrp_true)
np.save('RSRP/30_rsrp_pred' + str(today) + '.npy', rsrp_pred)
# 全部误差，
rsrp_err = abs(rsrp_pred - rsrp_true)
# # 单个id误差
# idx_error = np.mean(rsrp_err, axis=0)
# # 单个数据误差，
# sample_error = np.mean(rsrp_err, axis=1)

# 最优RSRP误差
rsrp_err = abs(rsrp_true.max(axis=1) - rsrp_pred.max(axis=1))
print('RSRP预测误差 = ', np.mean(rsrp_err))
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

print("new top-n")
rsrp_true=y_test
N = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rsrp_best_id_1 = rsrp_true[:, 0:32].argmax(1)
rsrp_best_id_2 = rsrp_true[:, 32 * 1:32 * (1 + 1)].argmax(1)
rsrp_best_id_3 = rsrp_true[:, 32 * 2:32 * (2 + 1)].argmax(1)
rsrp_best_id_4 = rsrp_true[:, 32 * 3:32 * (3 + 1)].argmax(1)
num = []
for i in N:
    id_n_1 = top_n(rsrp_pred[:, 0:32], i)
    id_n_2 = top_n(rsrp_pred[:, 32 * 1:32 * (1 + 1)], i)
    id_n_3 = top_n(rsrp_pred[:, 32 * 2:32 * (2 + 1)], i)
    id_n_4 = top_n(rsrp_pred[:, 32 * 3:32 * (3 + 1)], i)
    ans = 0
    for j in range(TEST_SMP):
        if rsrp_best_id_2[j] in id_n_2[j] :
            ans += 1
    num.append(ans)
for i in range(len(num)):
        # print('前' + str(i + 1) + '个预测结果中选择正确波速对的数目为：' + str(num[i]) + ' 概率为: ', num[i] / TEST_SMP)
    print(num[i] / TEST_SMP)