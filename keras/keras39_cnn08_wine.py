# 와인을 감정하는 데이터 //     # 다중분류 
import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)    #   (178, 13) (178,)
# print(y)
print(np.unique(y))            # 라벨의 unique 한 값 //  y는[0 1 2]만 있다.
print(np.unique(y, return_counts=True)) #   (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, 
    random_state=1000, test_size=0.2)

scaler = MinMaxScaler()
# scaler =StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape)            # (142, 13)
print(x_test.shape)             # (36, 13)

x_train = x_train.reshape(142, 13, 1, 1)
x_test = x_test.reshape(36, 13, 1, 1)

# #2. 모델구성
model = Sequential()
model.add(Conv2D(50, (2,1), input_shape=(13, 1, 1)))
model.add(Dense(20, activation='linear'))
model.add(Flatten())
model.add(Dense(3, activation='linear'))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min',patience=100, 
                  restore_best_weights=True,              
                   verbose=1)
import datetime                                             # 데이터 형식으로
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose = 1,
                      save_best_only=True,
                #      filepath = path + 'MCP/keras30_ModelCheckPoint3.hdf5')
                      filepath = filepath + 'k39_08_' + date +'_'+ filename) 
model.compile(loss='categorical_crossentropy', optimizer='adam',    # sparse_categorical_crossentropy 로 변경해도 가능
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1500, batch_size=32,
          callbacks=[es,mcp],
          validation_split=0.2,
          verbose=1)

#4. 평가, 검증
from sklearn.metrics import accuracy_score
import numpy as np

y_predict =  model.predict(x_test)
# print(y_predict)
y_predict = np.argmax(y_predict, axis = 1)                  # 가장 큰 자릿값 뽑아냄   / axis=1 (가로축(행)), axis=0 (세로축(열))
print("y_pred(예측값) : ", y_predict)

y_test = np.argmax(y_test, axis=1)
print("y_test(원래값) : ", y_test)

acc = accuracy_score(y_test, y_predict)                     # 소수점 들어가는 실수 형태로 구성// error 발생
print(acc)

# 0.75