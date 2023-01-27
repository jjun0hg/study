# 다중분류
import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout,LSTM
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) # (1797, 64)    (1797, )    // 64 = 8*8*1(흑백)
print(np.unique(y, return_counts=True))
#   (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),       // y에 들어갈 칼럼은 하나지만 o_h encoding 하면 칼럼(열) 10개로 늘어난다.
#   array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))          // 다중분류데이터(이미지 연산) // DNN방식
#   return_counts=True >> 각 숫자가 들어가는(나오는??) 횟수         // False 어떤 숫자가 나오는지

# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(datasets.images[4])
# plt.show()

y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, 
    random_state=333, test_size=0.2)

scaler = MinMaxScaler()
# scaler =StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape)        #(1437, 64)
print(x_test.shape)         #(360, 64)

x_train = x_train.reshape(1437, 16, 4)
x_test = x_test.reshape(360, 16, 4)

#2. 모델구성

model = Sequential()
model.add(LSTM(units=350, input_shape=(16,4), 
               return_sequences=True))
model.add(LSTM(units=250))
model.add(Dense(100, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(10))
model.summary()


#3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss', mode='min',patience=30, 
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
                      filepath = filepath + 'k48_09_' + date +'_'+ filename) 

model.compile(loss='categorical_crossentropy', optimizer='adam',    # sparse_categorical_crossentropy 로 변경해도 가능
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1, batch_size=32,
          callbacks=[es,mcp],
          validation_split=0.2,
          verbose=1)

#4. 평가, 검증
y_predict =  model.predict(x_test)
# print(y_predict)
y_predict = np.argmax(y_predict, axis = 1)                 
print("y_pred(예측값) : ", y_predict)

y_test = np.argmax(y_test, axis=1)
print("y_test(원래값) : ", y_test)

acc = accuracy_score(y_test, y_predict)
print(acc)      #   0.18055555555555555


