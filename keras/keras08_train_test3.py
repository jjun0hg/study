import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])    # (10, )
y = np.array(range(10))                 # (10, )

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size = 0.7, 
    # test_size = 0.3, 
    shuffle=True,                                   #False일 경우 test1, 2와 동일한 결과, shuffle작성하지 않으면 True로 default 되어있다.   
    random_state=123    
)

print('x_train : ', x_train)
print('x_test: :', x_test)
print('y_train : ', y_train)
print('y_test : ', y_test)

# x_train = x[:7]
# x_test = x[7:]
# y_train = y[:-3]
# # y_test = y[-3:]
# print(y_train)
# print(y_test)

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
shuffle 결과:
[ 8 10  1  2  6  7  3]
[4 5 9]
[7 9 0 1 5 6 2]
[3 4 8]

loss :  0.3423522412776947
[11]의 결과 :  [[10.709346]]

shuffle 결과:
[2 5 7 4 9 8 3]
[10  1  6]
[1 4 6 3 8 7 2]
[9 0 5]

loss :  0.5538934469223022
[11]의 결과 :  [[8.815238]]

shuffle 결과:
[10  2  6  7  8  4  3]
[9 1 5]
[9 1 5 6 7 3 2]
[8 0 4]

loss :  0.03756191208958626
[11]의 결과 :  [[10.050311]]

"""

