import tensorflow as tf
# import tensorflow
print(tf.__version__)
import numpy as np
# import numpy

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential #순차적인
from tensorflow.keras.layers import Dense
# y = wx +b 를 구현하기위한 import

model = Sequential()
model.add(Dense(1, input_dim=1))    # input_dim = 차원 (x = np.array([1,2,3]))  Dense(y, x) (덩어리개념)
                                    # 1은 y를 뜻함

#3. 컴파일과 훈련
model.compile(loss='mae', optimizer='adam')   # loss값을 낮추기 위한 기준을 'mae'로 한다. 'mae'를 최적화 하는 'adam'
model.fit(x, y, epochs=1000) # 초기 random값이 다르기 때문에 훈련할때마다 loss값 바뀐다.

#4. 평가, 예측
result = model.predict([4])
print('결과 : ', result)




