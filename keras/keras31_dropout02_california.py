# R2 0.55~0.6 이상
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, 
    random_state=123, test_size=0.2)


scaler = MinMaxScaler()
# scaler =StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# #2. 모델구성
# model = Sequential()
# model.add(Dense(10, input_shape=(8,), activation = 'relu'))
# model.add(Dense(20, activation ='relu'))
# model.add(Dense(30, activation ='relu'))
# model.add(Dense(40, activation ='relu'))
# model.add(Dense(20, activation ='relu'))
# model.add(Dense(10, activation ='relu'))
# model.add(Dense(10, activation ='relu'))
# model.add(Dense(1, activation = 'linear'))

#2. 모델구성(함수형 Model, input)
# Input_dim=13(열,특성,feature), output = 1
input1  = Input(shape=(8,))
dense1 = Dense(10, activation = 'relu')(input1)
dense2 = Dense(20, activation = 'relu')(dense1)
drop1 = Dropout(0.5)(dense2)
dense3 = Dense(30, activation = 'relu')(drop1)
dense4 = Dense(40, activation = 'relu')(dense3)
drop2 = Dropout(0.3)(dense4)
dense5 = Dense(20, activation = 'relu')(drop2)
dense6 = Dense(10, activation = 'relu')(dense5)
drop3 = Dropout(0.2)(dense6)
dense7 = Dense(10, activation = 'relu')(drop3)
output1 = Dense(1, activation = 'linear')(dense7)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min',patience=1600, 
                  restore_best_weights=True,              
                   verbose=1)
import datetime                                             # 데이터 형식으로
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose = 1,
                      save_best_only=True,
                #      filepath = path + 'MCP/keras30_ModelCheckPoint3.hdf5')
                      filepath = filepath + 'k31_02_' + date +'_'+ filename)

model.fit(x_train, y_train, epochs=350, batch_size=32,
              validation_split=0.25,
              callbacks=[es,mcp],
              verbose=1)



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)                           #   test 데이터로 평가
print('loss : ', loss)

y_predict = model.predict(x_test)                               #   최적의 가중치
from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))       
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)      



"""


"""



