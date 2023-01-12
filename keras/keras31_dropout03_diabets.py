# [과제, 실습]
# R2 0.62이상
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets['target'] 
##  333 x     123만 쓰기
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, 
    random_state=123, test_size=0.1)


scaler = MinMaxScaler()          
# scaler = StandardScaler()            
# # scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# # 2. 모델구성
# model = Sequential()
# model.add(Dense(200, input_shape=(10,), activation = 'relu'))
# model.add(Dense(90, activation='relu'))
# model.add(Dense(50,activation='relu'))
# model.add(Dense(8,activation='relu'))
# model.add(Dense(1, activation = 'linear'))

#2. 모델구성(함수형 Model, input)
# Input_dim=13(열,특성,feature), output = 1

inputs = Input(shape=(10,))
hidden1 = Dense(256, activation='linear') (inputs)
hidden2 = Dense(128) (hidden1)
drop1 = Dropout(0.5)(hidden2)
hidden3 = Dense(64) (drop1)
hidden4 = Dense(32) (hidden3)
drop2 = Dropout(0.3)(hidden4)
hidden5 = Dense(16) (drop2)
drop3 = Dropout(0.3)(hidden5)
hidden6 = Dense(8) (drop3)
output = Dense(1) (hidden6)
model = Model(inputs=inputs, outputs=output)



# 3. 컴파일, 훈련
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', mode='min',patience=16, 
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
                      filepath = filepath + 'k31_03_' + date +'_'+ filename)

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=10,
                callbacks=[es,mcp],
                validation_split=0.25)

# 4. 평가, 예측
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

