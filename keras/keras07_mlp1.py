# mlp = multy layer perceptron
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])
y = np.array([2,4,6,8,10,12,14,16,18,20])

print(x.shape)  #   (2,10)      shape=데이터 구조
print(y.shape)  #   (10,)

x = x.T         # 행과 열을 변환하기 위해 (T=transfer)
print(x.shape)  #   (10,2)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=2))        # input_dim = 열(컬럼, 피쳐, 특성)의 개수와 같다. 행무시 열우선 
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=4)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([[10, 1.4]])
print('[10, 1.4]의 예측값 :', result)


"""
결과 :
batch_size=1
10/10
loss :  0.10407407581806183
[10, 1.4]의 예측값 : [[19.957644]]

batch_size=2
5/5
loss :  0.12363366782665253
[10, 1.4]의 예측값 : [[20.209955]]

batch_size=3
4/4
loss :  0.2545686364173889
[10, 1.4]의 예측값 : [[19.643312]]

batch_size=4
3/3
loss :  0.10171103477478027
[10, 1.4]의 예측값 : [[20.06551]]

batch_size=5
2/2
loss :  0.12642356753349304
[10, 1.4]의 예측값 : [[19.750757]]

batch_size=6
2/2 
0.022448325529694557
[10, 1.4]의 예측값 : [[20.02831]]

batch_size=7
2/2 
loss :  0.2610591650009155
[10, 1.4]의 예측값 : [[19.686293]]

batch_size=8
2/2 
loss :  0.5922717452049255
[10, 1.4]의 예측값 : [[19.30611]]

batch_size=9
2/2 
loss :  0.5542938113212585
[10, 1.4]의 예측값 : [[19.18334]]

batch_size=10
1/1 
loss :  0.9005987048149109
[10, 1.4]의 예측값 : [[18.461634]]

"""
