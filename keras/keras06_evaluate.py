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
model.fit(x, y, epochs=10, batch_size=3)

#4. 평가, 예측
loss = model.evaluate(x, y)     # x, y만 현재 존재함, loss값 반환
                                # evalutate에 들어가는 값은 훈련 데이터에 들어가면 안된다.
                                # 판단의 기준은 loss로 한다.
                                # 통상적으로 predict로 판단하지 않는다.(예외 상황에서는 사람이 직접 확인해야함)
print('loss : ', loss)
result = model.predict([6])
print('6의 결과 :' , result)

# input, output은 바꿀수없다.
# 훈련량, 레이어의 깊이(개수) 조정 가능

