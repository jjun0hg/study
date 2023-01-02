import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6])     # 전체작업을 할지 부분(batch)을 나눠서 작업을 할지
y = np.array([1,2,3,5,4,6])

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))    # input layer를 뺀 나머지는 명시하지않는다.
model.add(Dense(1))     # 하이퍼 파라미터 튜닝이라고 한다.

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=10, batch_size=6)

#4. 평가, 예측
result = model.predict([6])
print('6의 결과 :' , result)

# input, output은 바꿀수없다.
# 훈련량, 레이어의 깊이(개수) 조정 가능

"""
batch_size=1
6/6 Epoch

batch_size=2
3/3 Epoch

batch size=3
2/2 Epoch

batch size=4
2/2 Epoch

batch size=5
2/2 Epoch

batch size=6
1/1 Epoch

batch size defalut 값은 32
"""
