from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from model import *
from utils import *
import datetime
import sys
import os
from sklearn.utils import shuffle
# os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
p = datetime.datetime.now()
today = p.strftime('%m_%d_%H_%M')
start_time = datetime.datetime.now()
sys.stdout = Logger('Log/' + str(today) + '.txt')
# sys.stdout = Logger('Log/427.txt')
tx_idx_4 = [6,9,22,25]

val_split_rate = 0.05
test_split_rate = 0.05



data3_all = np.load(r'data/user/oppodata/beam_rsrp_all_30.npy')


data9_all = np.load(r'data/user/oppodata/beam_rsrp_all_90.npy')



for i in range(25-1):
    if i==0:
        x3=data3_all[:,4*i:4*i+4,:,:]
        y3_f1=data3_all[:,4*(i+1),:,:]
        y3_f2=data3_all[:,4*(i+1)+1,:,:]
        y3_f3=data3_all[:,4*(i+1)+2,:,:]
        y3_f4=data3_all[:,4*(i+1)+3,:,:]
        #y_all=np.concatenate((y_f1,y_f2,y_f3,y_f4),axis=1)
    else:
        x3=np.concatenate((x3,data3_all[:,4*i:4*i+4,:,:]),axis=0)
        y3_f1=np.concatenate((y3_f1,data3_all[:,4*(i+1),:,:]),axis=0)
        y3_f2=np.concatenate((y3_f2,data3_all[:,4*(i+1)+1,:,:]),axis=0)
        y3_f3=np.concatenate((y3_f3,data3_all[:,4*(i+1)+2,:,:]),axis=0)
        y3_f4=np.concatenate((y3_f4,data3_all[:,4*(i+1)+3,:,:]),axis=0)
        y3_all=np.concatenate((y3_f1,y3_f2,y3_f3,y3_f4),axis=1)
        #y_all=np.concatenate((y_all,y_all_temp),axis=0)
for i in range(25-1):
    if i==0:
        x9=data9_all[:,4*i:4*i+4,:,:]
        y9_f1=data9_all[:,4*(i+1),:,:]
        y9_f2=data9_all[:,4*(i+1)+1,:,:]
        y9_f3=data9_all[:,4*(i+1)+2,:,:]
        y9_f4=data9_all[:,4*(i+1)+3,:,:]
        #y_all=np.concatenate((y_f1,y_f2,y_f3,y_f4),axis=1)
    else:
        x9=np.concatenate((x9,data9_all[:,4*i:4*i+4,:,:]),axis=0)
        y9_f1=np.concatenate((y9_f1,data9_all[:,4*(i+1),:,:]),axis=0)
        y9_f2=np.concatenate((y9_f2,data9_all[:,4*(i+1)+1,:,:]),axis=0)
        y9_f3=np.concatenate((y9_f3,data9_all[:,4*(i+1)+2,:,:]),axis=0)
        y9_f4=np.concatenate((y9_f4,data9_all[:,4*(i+1)+3,:,:]),axis=0)
        y9_all=np.concatenate((y9_f1,y9_f2,y9_f3,y9_f4),axis=1)
        #y_all=np.concatenate((y_all,y_all_temp),axis=0)
        
x_mix=np.concatenate((x3,x9),axis=0)
y_mix=np.concatenate((y3_all,y9_all),axis=0)

#用30km/h和90km/h混合训练
# x=x_min.reshape(50400,4,32,4,1)
# y=y3_all.reshape(50400,128*4)
# x, y = shuffle(x, y, random_state=42)
# x = x[:25200,:, tx_idx_4,:]
# y = y[:25200]

#只用90km/h训练
# x=x9.reshape(25200,4,32,4,1)
# y=y9_all.reshape(25200,128*4)

#只用30km/h训练
x=x3.reshape(25200,4,32,4,1)
y=y3_all.reshape(25200,128*4)
x, y = shuffle(x, y, random_state=42)
x = x[:,:, tx_idx_4,:]

SMP=25200

TEST_SMP = int(SMP * test_split_rate)
VAL_SMP = int(SMP * val_split_rate)

x_train = x[:SMP - TEST_SMP - VAL_SMP]
x_val = x[SMP - TEST_SMP - VAL_SMP:SMP - TEST_SMP]
x_test = x[SMP - TEST_SMP:]


y_train = y[:SMP - TEST_SMP - VAL_SMP]
y_val = y[SMP - TEST_SMP - VAL_SMP:SMP - TEST_SMP]
y_test = y[SMP - TEST_SMP:]

print(x_train.shape)
print(y_train.shape)
seed_tensorflow(42)
# seed_tensorflow(50)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    BM_input = keras.Input(shape=(4,4,4,1))
    BM_RSRP_output = LSTM_net(BM_input)
    BM_RSRP_NET = keras.Model(inputs=(BM_input), outputs=BM_RSRP_output, name='BM_RSRP_NET')
    # BM_RSRP_NET.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss=mse_best_n_id_loss)
    BM_RSRP_NET.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')
    BM_RSRP_NET.summary()

# callbacks
early_stopping = EarlyStopping(monitor='loss', mode='min', verbose=2, patience=45)
reducelronplateau = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=20, verbose=2, mode='auto', cooldown=0,
                                      min_lr=0)
best_model_path = "oppotime/models/60TX4T" + str(4) + "RX" + str(4) + "_" + str(today) + ".h5"
modelcheckpoint = ModelCheckpoint(best_model_path, save_weights_only=True, monitor='val_loss', mode='min', verbose=2,
                                  save_best_only=True)
callbacks_list = [early_stopping, modelcheckpoint, reducelronplateau]

# model training
hist = BM_RSRP_NET.fit(x=x_train, y=y_train, shuffle=True, batch_size=128, epochs=500, verbose=2,validation_data=(x_val,y_val),
                       callbacks=callbacks_list)
hist.history
loss = []
val_loss = []
loss.append(hist.history['loss'])
val_loss.append(hist.history['val_loss'])
np.save('oppotime/loss/loss' + str(today) + '.npy', loss)
np.save('oppotime/loss/val_loss' + str(today) + '.npy', val_loss)
# model test
rsrp_true = y_test

BM_RSRP_NET.load_weights(best_model_path)

# TEST
rsrp_pred = BM_RSRP_NET.predict(x_test)
rsrp_err = abs(rsrp_pred - rsrp_true)
print('全部RSRP预测误差 = ', np.mean(rsrp_err))

np.save('oppotime/RSRP/30_rsrp_true' + str(today) + '.npy', rsrp_true)
np.save('oppotime/RSRP/30_rsrp_pred' + str(today) + '.npy', rsrp_pred)
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
    #print('前' + str(i + 1) + '个预测结果中选择正确波速对的数目为：' + str(num[i]) + ' 概率为: ', num[i] / TEST_SMP)
    print(num[i] / TEST_SMP)

end_time = datetime.datetime.now()
total_time = end_time - start_time

def top_n(data, n):
    '''
    :param rec_list:
    :param n:
    :return:
    '''
    id_sort = np.argsort(data)
    id = id_sort[:, -n:]
    return id


for m in range(4):
    N = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    rsrp_best_id = rsrp_true[:, 128 * m:128 * (m + 1)].argmax(1)
    num = []
    for i in N:
        id_n = top_n(rsrp_pred[:, 128 * m:128 * (m + 1)], i)
        ans = 0
        for j in range(TEST_SMP):
            if rsrp_best_id[j] in id_n[j]:
                ans += 1
        num.append(ans)
    for i in range(len(num)):
        # print('前' + str(i + 1) + '个预测结果中选择正确波速对的数目为：' + str(num[i]) + ' 概率为: ', num[i] / TEST_SMP)
        print(num[i] / TEST_SMP)
print("new top-n")
N = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rsrp_best_id_1 = rsrp_true[:, 0:128].argmax(1)
rsrp_best_id_2 = rsrp_true[:, 128 * 1:128 * (1 + 1)].argmax(1)
rsrp_best_id_3 = rsrp_true[:, 128 * 2:128 * (2 + 1)].argmax(1)
rsrp_best_id_4 = rsrp_true[:, 128 * 3:128 * (3 + 1)].argmax(1)
num = []
for i in N:
    id_n_1 = top_n(rsrp_pred[:, 0:128], i)
    id_n_2 = top_n(rsrp_pred[:, 128 * 1:128 * (1 + 1)], i)
    id_n_3 = top_n(rsrp_pred[:, 128 * 2:128 * (2 + 1)], i)
    id_n_4 = top_n(rsrp_pred[:, 128 * 3:128 * (3 + 1)], i)
    ans = 0
    for j in range(TEST_SMP):
        if rsrp_best_id_1[j] in id_n_1[j] and rsrp_best_id_2[j] in id_n_2[j]:
            ans += 1
    num.append(ans)
for i in range(len(num)):
        # print('前' + str(i + 1) + '个预测结果中选择正确波速对的数目为：' + str(num[i]) + ' 概率为: ', num[i] / TEST_SMP)
    print(num[i] / TEST_SMP)
print(end_time)
print('time used: ', total_time)
