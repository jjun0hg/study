# 10, 1.4, 0을 넣었을 때 20이 나오는지
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
             [9,8,7,6,5,4,3,2,1,0]])

y = np.array([2,4,6,8,10,12,14,16,18,20])

print(x.shape)  #   (2,10)      shape=데이터 구조
print(y.shape)  #   (10,)

x = x.T         # 행과 열을 변환하기 위해 (T=transfer)
print(x.shape)  #   (10,2)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=3))        # input_dim = 열(컬럼, 피쳐, 특성)의 개수와 같다. 행무시 열우선 
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=10)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([[10, 1.4, 0]])
print('[10, 1.4, 0]의 예측값 :', result)


"""

결과 :
batch_size=1
10/10
loss :  0.027347827330231667
[10, 1.4, 0]의 예측값 : [[19.942492]]

batch_size=2
5/5
loss :  0.0615108497440815
[10, 1.4, 0]의 예측값 : [[20.098679]]

batch_size=3
4/4
loss :  0.07512649148702621
[10, 1.4, 0]의 예측값 : [[19.90212]]

batch_size=4
3/3
loss :  2.3896591663360596
[10, 1.4, 0]의 예측값 : [[16.214975]]

batch_size=5
2/2
loss :  4.213892936706543
[10, 1.4, 0]의 예측값 : [[11.420429]]

batch_size=6
2/2
loss :  1.9134738445281982
[10, 1.4, 0]의 예측값 : [[16.56464]]

batch_size=7
2/2
loss :  0.24136793613433838
[10, 1.4, 0]의 예측값 : [[19.590967]]

batch_size=8
2/2
loss :  0.09061479568481445
[10, 1.4, 0]의 예측값 : [[19.854048]]

batch_size=9
2/2
loss :  0.2746116518974304
[10, 1.4, 0]의 예측값 : [[20.023762]]

batch_size=10
1/1
loss :  3.1932013034820557
[10, 1.4, 0]의 예측값 : [[13.4752865]]
"""

