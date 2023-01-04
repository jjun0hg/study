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
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)   #   test 데이터로 평가
print('loss : ', loss)

y_predict = model.predict(x)            #   x에 대하여 예측 
import matplotlib.pyplot as plt
plt.scatter(x, y)                       #   점20 개 (x,y)
plt.plot(x, y_predict, color='red')     #   선긋기
plt.show()                              #   점 20개를 찍고, x값과, y의 x값에 대한 예측값을 나타낸다.

                                        # loss 훈련데이터와 테스트 데이터를 구분하여 진행했기 때문에 마지막 loss값의 차이가 있다.
                                        # 보편적으로 test데이터의 loss값이 더 좋지 않다.



