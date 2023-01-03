import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])    # (10, )
# y = np.array(range(10))                 # (10, )
x_train = np.array([1,2,3,4,5,6,7])     # (7, ) 
x_test = np.array([8,9,10])             # (3, )
y_train = np.array(range(7))
y_test = np.array(range(7,10))


#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=1))
model.add(Dense(100))
model.add(Dense(150))
model.add(Dense(190))
model.add(Dense(300))
model.add(Dense(120))
model.add(Dense(80))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer = 'adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
result = model.predict([11])
print('[11]의 결과 : ', result)

"""


결과 :
loss :  0.8376644253730774
[11]의 결과 :  [[11.039432]]



"""

