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
tx_idx_6 = [2, 5, 14, 17, 26, 29]
tx_idx_8 = [2, 5, 10, 13, 18, 21, 26, 29]

val_split_rate = 0.05
test_split_rate = 0.05


# data4_all = np.load(r'/home/len/ReID_test/beam_time/data/30km/beam_rsrp_all.npy')
# data4_id = np.load(r'/home/len/ReID_test/beam_time/data/30km/beam_best_idx.npy')

data4_all = np.load(r'/home/len/ReID_test/beam_time/data/90km/beam_rsrp_all.npy')
data4_id = np.load(r'/home/len/ReID_test/beam_time/data/90km/beam_best_idx.npy')

# data4_all = np.load(r'/home/len/ReID_test/beam_time/data/120km/beam_rsrp_all.npy')
# data4_id = np.load(r'/home/len/ReID_test/beam_time/data/120km/beam_best_idx.npy')

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
        y_all=np.concatenate((y_f1,y_f2,y_f3,y_f4),axis=1)
        #y_all=np.concatenate((y_all,y_all_temp),axis=0)

x=x.reshape(25200,4,32,4,1)
#y=y_f4.reshape(25200,128)

#预测未来4个时刻
y=y_all.reshape(25200,128*4)

# #预测未来1个时刻
# y=y_f1.reshape(25200,128*1)

x, y = shuffle(x, y, random_state=42)
x = x[:,:, tx_idx_4,:]

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
best_model_path = "models/60TX4T" + str(4) + "RX" + str(4) + "_" + str(today) + ".h5"
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
np.save('loss/loss' + str(today) + '.npy', loss)
np.save('loss/val_loss' + str(today) + '.npy', val_loss)
# model test
rsrp_true = y_test

BM_RSRP_NET.load_weights(best_model_path)

# TEST
rsrp_pred = BM_RSRP_NET.predict(x_test)
rsrp_err = abs(rsrp_pred - rsrp_true)
print('全部RSRP预测误差 = ', np.mean(rsrp_err))

np.save('RSRP/90_rsrp_true' + str(today) + '.npy', rsrp_true)
np.save('RSRP/90_rsrp_pred' + str(today) + '.npy', rsrp_pred)
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
print(end_time)
print('time used: ', total_time)
