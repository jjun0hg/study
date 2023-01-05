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

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.9, shuffle=True, random_state=123    
)

print(x)
print(x.shape)  #(442, 10)
print(y)
print(y.shape)  #(442, )

# 2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=10))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam', 
              metrics=['mse'])                                  #   metrics는 훈련에 영향을 미치진 않지만 값은 나온다.
                                                                #   acc와 accuracy는 동일
                                                                #   error가 그 다음 가중치 갱신을 할때 영향을 미친다.(훈련에 영향을 미친다.)
model.fit(x_train, y_train, epochs=200, batch_size=32)           #   가중치 생성

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)                           #   test 데이터로 평가
print('loss : ', loss)

y_predict = model.predict(x_test)                               #   최적의 가중치

print("=============")
print(y_test)
print(y_predict)
print("=============")

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))       
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)      


"""
결과

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.9, shuffle=True, random_state=123  
model = Sequential()
model.add(Dense(100, input_dim=10))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(1))    

model.compile(loss='mae', optimizer='adam', 
              metrics=['mse'])

model.fit(x_train, y_train, epochs=200, batch_size=32)
R2 :  0.6475770225494405
R2 :  0.6381546354721448
R2 :  0.6531883993086136
R2 :  0.6522781600401861
R2 :  0.6529517571902224


"""

