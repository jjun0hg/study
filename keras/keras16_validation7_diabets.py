# [과제, 실습]
# R2 0.62이상
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234, test_size=0.2)

# print(x)
# print(x.shape)  #(442, 10)
# print(y)
# print(y.shape)  #(442, )

# 2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=10, activation = 'relu'))
model.add(Dense(90, activation='relu'))
model.add(Dense(60,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1, activation = 'linear'))

# 3. 컴파일, 훈련
import time
start = time.time()
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=32,
          validation_split=0.3)
end = time.time()
print("걸린시간 : ", end - start)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)                           #   test 데이터로 평가
print('loss : ', loss)

y_predict = model.predict(x_test)                               #   최적의 가중치

"""
print("=============")
print(y_test)
print(y_predict)
print("=============")
"""

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))       
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)      



"""



"""

