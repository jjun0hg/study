# RMSE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,21))
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123    
)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae'])                                  #   metrics는 훈련에 영향을 미치진 않지만 값은 나온다.
                                                                #   acc와 accuracy는 동일
                                                                #   error가 그 다음 가중치 갱신을 할때 영향을 미친다.(훈련에 영향을 미친다.)
model.fit(x_train, y_train, epochs=200, batch_size=1)           #   가중치 생성

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)                           #   test 데이터로 평가
print('loss : ', loss)

y_predict = model.predict(x_test)                               #   최적의 가중치

print("=============")
print(y_test)
print(y_predict)
print("=============")

# RMSE function
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))       #   np.sqrt = RMSE, mean_squared_error( , )= MSE

print("RMSE : ", RMSE(y_test, y_predict))

# RMSE :  3.86149851789604
# RMSE :  3.8505546021288635    가장 좋은 가중치
# RMSE :  3.855078208869394
# RMSE :  3.919321611530858







