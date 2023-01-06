import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))
x_train = x[:10]    # [ 1  2  3  4  5  6  7  8  9 10]
y_train = y[:10]    # [ 1  2  3  4  5  6  7  8  9 10]
x_test = x[10:13]   # [11 12 13]
y_test = y[10:13]   # [11 12 13]
x_validation =x[13:16]
y_validation =y[13:16]
# x_train = np.array(range(1,11))
# y_train = np.array(range(1,11))
# x_test = np.array([11,12,13])
# y_test = np.array([11,12,13])
# x_validation = np.array([14,15,16])
# y_validation = np.array([14,15,16])

#2. 모델
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation ='relu'))
model.add(Dense(1))

#3. 컴파일
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_validation, y_validation))

#4 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print("17의 예측값 : ", result)




