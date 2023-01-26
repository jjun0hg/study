import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime      

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],
             [4,5,6],[5,6,7],[6,7,8],
             [7,8,9],[8,9,10],[9,10,11],
             [10,11,12],[20,30,40],
             [30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])
x = x.reshape(13,3,1)
#2. 모델구성

model = Sequential()
model.add(LSTM(units=700, input_shape=(3,1)))
model.add(Dropout(0.3))
model.add(Dense(100, activation='linear'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='linear'))
model.add(Dense(1))
model.summary()

#3. 컴파일, 훈련

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
                        filepath = filepath + 'k45_01_' + date +'_'+ filename)
model.fit(x,y, epochs=1000, batch_size=5,
          callbacks=[es,mcp],verbose=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
print(loss)
y_pred = x_predict.reshape(1,3,1)
result = model.predict(y_pred)
print('[50,60,70]의 결과 : ', result )
# [50,60,70]의 결과 :  [[71.27633]]







