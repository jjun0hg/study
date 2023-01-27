import numpy as np
a = np.array(range(1,101))
x_predict = np.array(range(96, 106))    
# 예상 y = 100, 107

timesteps = 5   # x는 4개, y는 1개

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, timesteps)
x = bbb[:, :-1]
y = bbb[:, -1]
x = x.reshape(96,4*1)

# print(x,y)

timesteps = 4   # y는 없다.
def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

x_predict = split_x(x_predict, timesteps)
# print(x_predict)
# print(x_predict.shape)

x_predict = x_predict.reshape(7,4,1)

# 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime 

model = Sequential()
model.add(Dense(128, activation = 'relu', input_shape=(4,)))    
model.add(Dense(64, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(1))
model.summary()

"""

train_test_split = 2차원만 받아들임
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=1234
)

"""

# 컴파일, 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es = EarlyStopping(monitor='loss', mode='min',patience=3, 
                  restore_best_weights=True,                
                   verbose=1)

filepath = './_save/MCP/'
filename = '{epoch:04d}-{loss:.4f}.hdf5'  
date = datetime.datetime.now()   
date = date.strftime("%m%d_%H%M") 
mcp = ModelCheckpoint(monitor='loss', mode = 'auto', verbose = 1,
                        save_best_only=True,
                        filepath = filepath + 'k47_3_' + date +'_'+ filename)
model.fit(x,y, epochs=100, batch_size=1,
          callbacks=[es,mcp],verbose=1)

# 평가, 예측
loss = model.evaluate(x,y)
print(loss)
result = model.predict(x_predict)
print('[100, 107]의 결과 : ', result )

# [100, 107]의 결과 :  [[ 99.68383 ][100.68564 ][101.68749 ][102.689316][103.69113 ][104.69296 ][105.6948  ]]


