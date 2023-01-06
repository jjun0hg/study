import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234, test_size=0.2)
print(x_train.shape, x_test.shape)  #(12, ) (4,)
print(y_train.shape, y_test.shape)  #(12, ) (4,)

# x_train = x[:10]    # [ 1  2  3  4  5  6  7  8  9 10]
# y_train = y[:10]    # [ 1  2  3  4  5  6  7  8  9 10]
# x_test = x[10:13]   # [11 12 13]
# y_test = y[10:13]   # [11 12 13]
# x_val =x[13:16]
# y_val =y[13:16]

#2. 모델
model = Sequential()
model.add(Dense(5, input_dim=1, activation = 'linear'))
model.add(Dense(10, activation ='relu'))
model.add(Dense(5, activation ='relu'))
model.add(Dense(1, activation = 'linear'))

#3. 컴파일
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=1,
          validation_split=0.25)

#4 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print("17의 예측값 : ", result)




