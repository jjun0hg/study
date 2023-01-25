# 31_1 복붙
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
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

print(x_train.shape)

x_train = x_train.reshape(404, 13, 1, 1)
x_test = x_test.reshape(102, 13, 1, 1)

# 2. 모델구성

model = Sequential()
model.add(Conv2D(64, (2,1), input_shape=(13, 1, 1)))
model.add(Flatten())
model.add(Dense(1, activation='linear'))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', mode='min',patience=30, 
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
                      filepath = filepath + 'k39_01_' + date +'_'+ filename)

                      
model.fit(x_train, y_train, epochs=300, batch_size=8,
          callbacks=[es,mcp],verbose=1,
          validation_split=0.25)

# model.save(path + "keras30_ModelCheckPoint3_save_model.h5")

#4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse : ', mse)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)      

# r2스코어 :  0.8076792409881869
