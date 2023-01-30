import pandas as pd
import numpy as np

path = './_data/stock/'
samsung = pd.read_csv(path + '삼성전자 주가.csv',header=0, index_col=None, sep=',', encoding='CP949')
amore = pd.read_csv(path + '아모레퍼시픽 주가.csv',header=0, index_col=None, sep=',', encoding='CP949')

samsung.rename(columns = {"Unnamed: 6": "??"}, inplace = True)
samsung = samsung.drop(['전일비','등락률','??','금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램','외인비',],axis=1)

amore.rename(columns = {"Unnamed: 6": "??"}, inplace = True)
amore = amore.drop(['전일비','등락률','??','금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램','외인비',],axis=1)

# 결측치 제거
samsung =  samsung.dropna()
amore = amore.dropna()
# print(samsung.shape,amore.shape)    (1977, 6) (2210, 6)

# 오름차순 정렬
samsung = samsung.sort_values(['일자'], ascending=True)
amore = amore.sort_values(['일자'], ascending=True)

samsung.reset_index(drop=True,inplace=True)
amore.reset_index(drop=True,inplace=True)
samsung = samsung.drop('일자',axis=1)
amore = amore.drop('일자',axis=1)

samsung = samsung.drop(samsung.index[0:1700])
samsung.reset_index(drop=True,inplace=True) 
print(samsung.shape,amore.shape)    #(1877, 5) (2210, 5)

# shape 맞추기
amore = amore.drop(amore.index[0:1933])
amore.reset_index(drop=True,inplace=True) 
print(samsung.shape,amore.shape)    #(1877, 5) (1877, 5)
print(samsung)
# 숫자 ',' 제거
for i in range(len(samsung.index)):
    for k in range(len(samsung.iloc[i])):
        samsung.iloc[i,k] = int(samsung.iloc[i,k].replace(',',""))

for i in range(len(amore.index)):
    for k in range(len(amore.iloc[i])):
        amore.iloc[i,k] = int(amore.iloc[i,k].replace(',',""))

samsung = np.array(samsung)
amore = np.array(amore)
print(type(samsung),type(amore))

# x dataset
samsung_x = samsung[:,[1,2,3,4]]
samsung_y = samsung[:,0]
print(samsung_x)
amore_x = amore[:,[1,2,3,4]]
amore_y = amore[:,0]

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i+timesteps)]
        aaa.append(subset)
    return np.array(aaa)
samsung_x = split_x(samsung_x, 5)
amore_x = split_x(amore_x, 5)
# print(samsung_x.shape, amore_x.shape) (1873, 5, 4) (1873, 5, 4)

samsung_pred_test = samsung_x[-2]
amore_pred_test = amore_x[-2]

samsung_predict = samsung_x[-1]
amore_predict = amore_x[-1]

samsung_x=np.delete(samsung_x,[271],axis=0)
amore_x=np.delete(amore_x,[271],axis=0)

samsung_x=np.delete(samsung_x,[270],axis=0)
amore_x=np.delete(amore_x,[270],axis=0)

# print(samsung_x.shape, amore_x.shape) (1871, 5, 4) (1871, 5, 4)

# y dataset
samsung_y=np.delete(samsung_y,[0,1,2,3,4,5],axis=0) 
amore_y=np.delete(amore_y,[0,1,2,3,4,5],axis=0) 
print(samsung_y.shape, amore_y.shape)

# 1. 데이터
samsung_x = samsung_x.astype('float32')
samsung_y = samsung_y.astype('float32')
samsung_pred = samsung_predict.astype('float32')
amore_x = amore_x.astype('float32')
amore_y = amore_y.astype('float32')
amore_pred = amore_predict.astype('float32')
samsung_pred_test =  samsung_pred_test.astype('float32')
amore_pred_test =  amore_pred_test.astype('float32')


from sklearn.model_selection import train_test_split
samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test = train_test_split(
    samsung_x, samsung_y, train_size=0.8, random_state=1)
amore_x_train, amore_x_test, amore_y_train, amore_y_test = train_test_split(
    amore_x, amore_y, train_size=0.8, random_state=1)

samsung_pred = samsung_pred.reshape(1,samsung_pred.shape[0],samsung_pred.shape[1])
amore_pred = amore_pred.reshape(1,amore_pred.shape[0],amore_pred.shape[1])
print(samsung_pred.shape,amore_pred.shape)  # (1, 5, 4) (1, 5, 4)

from sklearn.preprocessing import StandardScaler

num_sample   = samsung_x_train.shape[0] # 샘플 데이터 수
num_sequence = samsung_x_train.shape[1] # 시계열 데이터 수 
num_feature  = samsung_x_train.shape[2] # Feature 수 
scaler = StandardScaler()
for ss in range(num_sequence):
    scaler.partial_fit(samsung_x_train[:, ss, :]) #fit은 train data만 함

# 1. samsung train data
results = []
for ss in range(num_sequence):
    results.append(scaler.transform(samsung_x_train[:, ss, :]).reshape(num_sample, 1, num_feature))
samsung_x_train = np.concatenate(results, axis=1)
# 2. samsung test data
num_sample   = samsung_x_test.shape[0] # 샘플 데이터 수 
num_sequence = samsung_x_test.shape[1] # 시계열 데이터 수
num_feature  = samsung_x_test.shape[2] # Feature 수 
results = []
for ss in range(num_sequence):
    results.append(scaler.transform(samsung_x_test[:, ss, :]).reshape(num_sample, 1, num_feature))
samsung_x_test = np.concatenate(results, axis=1)
# 3. samsung predict data
num_sample   = samsung_pred.shape[0] # 샘플 데이터 수
num_sequence = samsung_pred.shape[1] # 시계열 데이터 수
num_feature  = samsung_pred.shape[2] # Feature 수
results = []
for ss in range(num_sequence):
    results.append(scaler.transform(samsung_pred[:, ss, :]).reshape(num_sample, 1, num_feature))
samsung_pred = np.concatenate(results, axis=1)


num_sample   = amore_x_train.shape[0] # 샘플 데이터 수
num_sequence = amore_x_train.shape[1] # 시계열 데이터 수 
num_feature  = amore_x_train.shape[2] # Feature 수 
scaler = StandardScaler()
for ss in range(num_sequence):
    scaler.partial_fit(amore_x_train[:, ss, :]) #fit은 train data만 함
results = []
# 1. amore train data
for ss in range(num_sequence):
    results.append(scaler.transform(amore_x_train[:, ss, :]).reshape(num_sample, 1, num_feature))
amore_x_train = np.concatenate(results, axis=1)
# 2. amore test data
num_sample   = amore_x_test.shape[0] # 샘플 데이터 수 
num_sequence = amore_x_test.shape[1] # 시계열 데이터 수 
num_feature  = amore_x_test.shape[2] # Feature 수 
results = []
for ss in range(num_sequence):
    results.append(scaler.transform(amore_x_test[:, ss, :]).reshape(num_sample, 1, num_feature))
amore_x_test = np.concatenate(results, axis=1)
# 3. amore predict data
num_sample   = amore_pred.shape[0] # 샘플 데이터 수
num_sequence = amore_pred.shape[1] # 시계열 데이터 수
num_feature  = amore_pred.shape[2] # Feature 수
results = []
for ss in range(num_sequence):
    results.append(scaler.transform(amore_pred[:, ss, :]).reshape(num_sample, 1, num_feature))
amore_pred = np.concatenate(results, axis=1)

print(samsung_x.shape) #
print(samsung_y.shape)
print(amore.shape)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Sequential, Model, load_model
# model = load_model(path + 'stock0129_2354_0273-90703160.0000.hdf5')

#2-1. 모델1.
input1 = Input(shape=(5,4))
dense1 = Dense(300, activation='relu', name='ds11')(input1)
dense2 = Dropout(0.3)(dense1)
dense3 = Dense(500, activation='relu', name='ds12')(dense2)
dense4 = Dense(100, activation='relu', name='ds13')(dense3)
dense5 = Dense(20, activation='relu', name='ds14')(dense4)
output1 = Dense(10, activation='relu', name='ds15')(dense5)


#2-2. 모델2.
input2 = Input(shape=(5,4))
dense21 = Dense(300, activation='relu', name='ds21')(input2)
dense22 = Dense(500, activation='relu', name='ds22')(dense21)
dense23 = Dense(100, activation='relu', name='ds23')(dense22)
dense24 = Dense(20, activation='relu', name='ds24')(dense23)
output2 = Dense(10, activation='linear', name='ds25')(dense24)

#2-3. 모델병합
from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2], name='mg1')
merge2 = Dense(300, activation='relu', name='mg2')(merge1)
merge3 = Dense(50, name='mg3')(merge2)
merge4 = Dense(20, name='mg4')(merge3)
last_output = Dense(1, name='last')(merge4)

model = Model(inputs=[input1, input2], outputs = last_output)

model.summary()
import tensorflow as tf
optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min',patience=50, 
                  restore_best_weights=True,              
                   verbose=1)
import datetime                                             # 데이터 형식으로
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose = 1,
                      save_best_only=True,
                      filepath = filepath + 'stock' + date +'_'+ filename)
model.fit([samsung_x_train,amore_x_train], samsung_y_train, epochs=1, 
            verbose= 1, 
            batch_size=250,
            validation_split=0.33,
            callbacks=[es,mcp])

loss,mae = model.evaluate([samsung_x_test, amore_x_test], samsung_y_test,batch_size=32)
print("loss : ",loss)
print("mae : ",mae)

result = model.predict([samsung_pred,amore_pred])
print("삼성 시가예측값 : ", result)

