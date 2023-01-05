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
x_train, x_test, y_train, y_test = train_test_split(x, y,
                        train_size=0.7, shuffle=True, random_state=115)
# print(x)
# print(x.shape)  # (506, 13)
# print(y)
# print(y.shape)  # (506,)

print(dataset.feature_names)    # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
print(dataset.DESCR)

#2. 모델구성
# Input_dim=13(열,특성,feature), output = 1
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))


#3. 컴파일, 훈련
#loss = mae or mse optimizer= 'adam', matrix[mae or mse]
import time
start = time.time()
model.compile(loss='mse', optimizer='adam',
                metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=30)
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

x_train, x_test, y_train, y_test = train_test_split(x, y,
                        train_size=0.8, shuffle=True, random_state=123)


model = Sequential()
model.add(Dense(13, input_dim=13))
model.add(Dense(15))
model.add(Dense(17))
model.add(Dense(19))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(9))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))
R2 :  0.6170848099202683

x_train, x_test, y_train, y_test = train_test_split(x, y,
                        train_size=0.7, shuffle=True, random_state=115)
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(30))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(70))
model.add(Dense(1))
model.fit(x_train, y_train, epochs=400, batch_size=1)
R2 :  0.6357098369174583



x_train, x_test, y_train, y_test = train_test_split(x, y,
                        train_size=0.7, shuffle=True, random_state=115)

model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))
model.fit(x_train, y_train, epochs=50000, batch_size=30)

R2 :  0.7831901193099949
"""


