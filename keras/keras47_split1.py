import numpy as np

a = np.array(range(1,11))
timesteps = 5


def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, timesteps)

print(bbb)

x = bbb[:, :-1]
y = bbb[:, -1]
print(x, y)
print(x.shape, y.shape)     # (6, 4) (6, )
x = x.reshape(6,4,1)

# 실습
# LSTM 모델구성
x_predict = np.array([7, 8, 9, 10])

# 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime 

model = Sequential()
model.add(LSTM(units=64, input_shape=(4,1)))
model.add(Dense(32, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(1))
model.summary()

# 컴파일, 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es = EarlyStopping(monitor='loss', mode='min',patience=100, 
                  restore_best_weights=True,                
                   verbose=1)

filepath = './_save/MCP/'
filename = '{epoch:04d}-{loss:.4f}.hdf5'  
date = datetime.datetime.now()   
date = date.strftime("%m%d_%H%M") 
mcp = ModelCheckpoint(monitor='loss', mode = 'auto', verbose = 1,
                        save_best_only=True,
                        filepath = filepath + 'k47_' + date +'_'+ filename)
model.fit(x,y, epochs=1000, batch_size=5,
          callbacks=[es,mcp],verbose=1)

# 평가, 예측
loss = model.evaluate(x,y)
print(loss)
y_pred = x_predict.reshape(1,4,1)
result = model.predict(y_pred)
print('[7, 8, 9, 10]의 결과 : ', result )

# [7, 8, 9, 10]의 결과 :  [[10.981844]]
# model = Sequential()
# model.add(LSTM(units=64, input_shape=(4,1)))
# model.add(Dense(32, activation='linear'))
# model.add(Dense(10, activation='linear'))
# model.add(Dense(1))
# model.fit(x,y, epochs=1000, batch_size=5,
#   callbacks=[es,mcp],verbose=1)
