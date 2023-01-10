from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x= np.array([1,2,3])
y= np.array([1,2,3])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련 

model.summary()
# output shape(아키텍쳐 구조), param(아키텍쳐 연산량)
# bias== 계산량에 포함된다.

