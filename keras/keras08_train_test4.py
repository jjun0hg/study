# train_test_split를 이용하여 7:3으로 잘라서 모델 구현

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])       # 0부터 10-1까지
                                                                # print(range(10))              # ctrl + / 주석처리
                                                                # print(x.shape)      # (3,10)
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])
x = x.T
y = y.T
# print(x.shape)      #   (10,3)
# print(y.shape)      #   (10,2)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size = 0.7, 
    # test_size = 0.3, 
    shuffle=True,                                               #False일 경우 test1, 2와 동일한 결과, shuffle작성하지 않으면 True로 default 되어있다.   
    # random_state=123    
)

print('x_train : ', x_train)
print('x_test: :', x_test)
print('y_train : ', y_train)
print('y_test : ', y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(9, input_dim=3))                                # input_dim = 열(컬럼, 피쳐, 특성)의 개수와 같다. 행무시 열우선 
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
x_train :  [
            [  8  29 209]
            [  0  21 201]
            [  4  25 205]
            [  7  28 208]
            [  3  24 204]
            [  9  30 210]
            [  2  23 203]
]
x_test: : [
            [  5  26 206]
            [  1  22 202]
            [  6  27 207]
]
y_train :  [[ 9.   1.6]
            [ 1.   1. ]
            [ 5.   2. ]
            [ 8.   1.5]
            [ 4.   1. ]
            [10.   1.4]
            [ 3.   1. ]
]
y_test :  [
            [6.  1.3]
            [2.  1. ]
            [7.  1.4]
]

loss :  0.3409635126590729
[9, 30, 210]의 예측값 : [[9.960465  1.4794551]]

"""
