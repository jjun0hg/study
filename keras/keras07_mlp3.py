import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])       # 0부터 10-1까지
# print(range(10))              # ctrl + / 주석처리
print(x.shape)      # (3,10)
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])

x = x.T
y = y.T
print(x.shape)      #   (10,3)
print(y.shape)      #   (10,2)

#2. 모델구성
model = Sequential()
model.add(Dense(9, input_dim=3))        # input_dim = 열(컬럼, 피쳐, 특성)의 개수와 같다. 행무시 열우선 
model.add(Dense(15))
model.add(Dense(21))
model.add(Dense(24))
model.add(Dense(27))
model.add(Dense(30))
model.add(Dense(33))
model.add(Dense(27))
model.add(Dense(21))
model.add(Dense(15))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(2))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([[9, 30, 210]])
print('[9, 30, 210]의 예측값 :', result)

"""
결과값
loss :  0.22666311264038086
[9, 30, 210]의 예측값 : [[9.998872  1.6628157]]

"""