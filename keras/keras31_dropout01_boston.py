from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = './_save/'
# path = '../save/'
# path = 'c:/study/_save/'

#1. 데이터          /random_state = 123/ 1/ 365 /100000

datasets = load_boston()
x = datasets.data
y = datasets['target'] 


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, 
    random_state=100000, test_size=0.2
)
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# # 2. 모델구성(순차형/Sequential())
# Input_dim=13(열,특성,feature), output = 1
# model = Sequential()
# model.add(Dense(50, input_dim=13, activation = 'relu'))
# model.add(Dropout(0.5))
# model.add(Dense(40, activation = 'sigmoid'))
# model.add(Dropout(0.3))
# model.add(Dense(30, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(20, activation = 'relu'))
# model.add(Dense(1, activation = 'linear'))
# model.summary()

#2. 모델구성(함수형 Model, input)
# Input_dim=13(열,특성,feature), output = 1
# 훈련시에만 적용된다.

input1  = Input(shape=(13,))
dense1 = Dense(50, activation = 'relu')(input1)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(40, activation = 'sigmoid')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(30, activation = 'relu')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(20, activation = 'linear')(drop3)
output1 = Dense(1, activation = 'linear')(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', mode='min',patience=1600, 
                  restore_best_weights=True,                # False 일때 MCP 데이터가 높아야한다.(이론상으로)
                   verbose=1) 
import datetime                                             # 데이터 형식으로
date = datetime.datetime.now()                              # 현재 시간
# print(type(date))                                           # <class 'datetime.datetime'>  string 문자열 형태로 바꿔야함
date = date.strftime("%m%d_%H%M")                           # 0112_1457
# print(date)                                                 # <class 'str'>

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'                # [{epoch:04d} =epoch100 == 0100] - [{val_loss:.4f}= 소수4째자리] // 0037-0.0048.hdf5

mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose = 1,
                      save_best_only=True,
                #      filepath = path + 'MCP/keras30_ModelCheckPoint3.hdf5')
                      filepath = filepath + 'k31_01_' + date +'_'+ filename)

                      
model.fit(x_train, y_train, epochs=300, batch_size=8,
          callbacks=[es,mcp],verbose=1,
          validation_split=0.25)

# model.save(path + "keras30_ModelCheckPoint3_save_model.h5")

#4. 평가, 예측
print("=====================1. 기본 출력 ========================")
mse, mae = model.evaluate(x_test, y_test)
print('mse : ', mse)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)      

print("=====================2. load_model 출력 ========================")
model2 = load_model(path + "keras30_ModelCheckPoint3_save_model.h5")
mse, mae = model2.evaluate(x_test, y_test)
print('mse : ', mse)

y_predict = model2.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)      

# 통상적으로 MCP 사용
print("=====================3. ModelCheckPoint 출력 ========================")
model3 = load_model(path + "MCP/keras30_ModelCheckPoint3.hdf5")
mse, mae = model3.evaluate(x_test, y_test)
print('mse : ', mse)

y_predict = model3.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)

