# 1. train 0.7 이상
# 2. R2 : 0.8 이상 / RMSE 사용
"""
import sklearn as sk
print(sk.__version__)   # 1.1.3


"""
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target  #   가격 데이터
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, test_size=0.2)

# print(dataset.feature_names)    # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
# print(dataset.DESCR)

#2. 모델구성
# Input_dim=13(열,특성,feature), output = 1
model = Sequential()
model.add(Dense(26, input_dim=13, activation = 'linear'))
model.add(Dense(52, activation = 'relu'))
model.add(Dense(24, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))


#3. 컴파일, 훈련
#loss = mae or mse optimizer= 'adam', matrix[mae or mse]
import time
start = time.time()
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=800, batch_size=32,
          validation_split=0.25)
end = time.time()
print('걸린시간 : ', end - start)
# gpu 6.4
# cpu 1.7

#4. 평가, 예측
# model.fit(x_train, y_train) epoch batch_size=32(default)
# model.evaluate(x_test, y_test)
# y_predict
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


"""

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234, test_size=0.2)
model.add(Dense(5, input_dim=13, activation = 'linear'))
model.add(Dense(10, activation ='relu'))
model.add(Dense(5, activation ='relu'))
model.add(Dense(1, activation = 'linear'))
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.25)
R2 :  0.7016786262095126



x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, test_size=0.2)

model = Sequential()
model.add(Dense(26, input_dim=13, activation = 'linear'))
model.add(Dense(52, activation = 'relu'))
model.add(Dense(24, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))
0.7276881435330604

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=800, batch_size=32,
          validation_split=0.25)
"""


