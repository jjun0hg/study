import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터

dataset = np.array([1,2,3,4,5,6,7,8,9,10]) # 시계열 데이터는 y가 없다.. // (10, )

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
             [5,6,7], [6,7,8], [7,8,9]])       

y = np.array([4,5,6,7,8,9,10])                  

print(x.shape, y.shape)     # (7, 3) (7,)

x = x.reshape(7,3,1)
print(x.shape)              # (7, 3, 1)

#2. 모델구성

model = Sequential()
# model.add(SimpleRNN(units = 200, input_shape=(3,1)))
                            # (N, 3, 1) -> ([batch, timesteps, feature])
# model.add(SimpleRNN(units=64, input_length=3, input_dim=1))
# model.add(SimpleRNN(units=64, input_dim=1, input_length=3))     # 가독성 떨어짐
# model.add(LSTM(units=10, input_shape=(3,1)))

model.add(GRU(units=10, input_shape=(3,1)))
model.add(Dropout(0.3))
model.add(Dense(400, activation='relu'))
# model.add(Dropout(0.4))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary()

# SimpleRNN == param = 10 * (10 + 1 + 1) = 120
# LSTM == param = (4 * {(10) * (10 + 1 + 1)}) = 480
# GRU == 
# units * ( feature + bias + units) = params




