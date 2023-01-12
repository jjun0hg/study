# 다중분류
import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

#1. 데이터
datasets = fetch_covtype()      # sklearn 데이터 셋 안에 numpy로 저장되어 있다.
x = datasets.data               # numpy로 변환 되는 것이 아니다.
y = datasets['target']
print(type(x))
# print(x.shape, y.shape)     #   (581012, 54)   (581012,)
# print(np.unique(y, return_counts=True))     #   (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))
print(y.shape)

###################################################인코딩##############################################################
# get_dummies       // .values or numpy 로 변환 // index, header 자동생성 // numpy 자료형이 pandas를 바로 못받아들임
y = pd.get_dummies(y, drop_first=False)
# # y = y.values
# # y = y.to_numpy()
y = np.array(y)

# to_categorical
# y = to_categorical(y)
# print(type(y))
# # print(y[:10])
# # print(np.unique(y[:,0], return_counts=True))            # 모든 행의 0번째
# y = np.delete(y, 0, axis=1)

# OneHotEncoder     preprocessing = 전처리 // sparse=True default ==> Matrix 반환 // array가 필요하므로 False // y = y.toarray()
# Ohe = OneHotEncoder(sparse=True)
# # y = y.reshape(581012, 1)
# y = y.reshape(-1, 1)
# # # print(y.shape)
# # # Ohe.fit(y)                  
# # # print(y.shape)
# # y = Ohe.transform(y)            # 원하는 형식으로 변환 // 훈련시킨 결과(영향)에 대한 생성
# y = Ohe.fit_transform(y)
# y = y.toarray()
# print(y[:15])
# print(type(y))        scipy.sparse.~_maxtrix
# sparse = True .toarray()      // False 그대로

######################################################################################################################

# num = np.unique(datasets['target'], axis=0)
# num = num.shape[0]
# encoding = np.eye(num)[datasets['target']]
# y = encoding
# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, 
    random_state=333, test_size=0.2)

# scaler = MinMaxScaler()           #   특성들을 특정 범위(주로 [0,1]) 로 스케일링 하는 것  // 이상치에 매우 민감하다
scaler =StandardScaler()            #   최솟값과 최댓값의 크기를 제한하지 않기 때문에, 어떤 알고리즘에서는 문제가 있을 수 있으며 이상치에 매우 민감하다
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# #2. 모델구성
# model = Sequential()
# model.add(Dense(100, activation='relu', input_shape=(54,)))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='linear'))
# model.add(Dense(7, activation='softmax'))       # 열은 하나인데 칼럼 7개

#2. 모델구성
input1  = Input(shape=(54,))
dense1 = Dense(100, activation = 'relu')(input1)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(50, activation = 'relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(30, activation = 'relu')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(20, activation = 'relu')(drop3)
dense5 = Dense(10, activation = 'linear')(dense4)
output1 = Dense(7, activation = 'softmax')(dense5)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss', mode='min',patience=16, 
                  restore_best_weights=True,              
                   verbose=1)
import datetime                                             # 데이터 형식으로
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath = './_save/MCP/'
filename = '{epoch:04d}-{loss:.4f}.hdf5'
mcp = ModelCheckpoint(monitor='loss', mode = 'auto', verbose = 1,
                      save_best_only=True,
                #      filepath = path + 'MCP/keras30_ModelCheckPoint3.hdf5')
                      filepath = filepath + 'k31_10_' + date +'_'+ filename) 

model.compile(loss='categorical_crossentropy', optimizer='adam',    # sparse_categorical_crossentropy 로 변경해도 가능
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=64,
        callbacks=[es,mcp],
        # validation_split=0.2,
        verbose=1)

# print(type(y))

#4. 평가, 검증
y_predict =  model.predict(x_test)
# print(y_predict)
y_predict = np.argmax(y_predict, axis = 1)                 
print("y_pred(예측값) : ", y_predict[:20])

y_test = np.argmax(y_test , axis=1)
print("y_test(원래값) : ", y_test[:20])

acc = accuracy_score(y_test, y_predict)
print(acc)      #   0.9116201819230141

