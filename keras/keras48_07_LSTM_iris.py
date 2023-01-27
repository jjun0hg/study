# 다중분류  //categorical_crossentropy
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, LSTM
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd

#1. 데이터
datasets = load_iris()
# print(datasets.DESCR)             # pandas .describe() // .info()
# print(datasets.feature_names)     # pandas.columns

x = datasets.data
y = datasets['target'] # datasets.target
# print(x.shape, y.shape)           # (150, 4) (150,)

# 인코딩

# num = np.unique(datasets['target'], axis=0)
# num = num.shape[0]
# encoding = np.eye(num)[datasets['target']]
# y = encoding
# print(y.shape)

# y = pd.get_dummies(y, drop_first=False)
# y = np.array(y)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
# print(y.shape)        // (150,3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, 
    random_state=333, test_size=0.2)

scaler = MinMaxScaler()
# scaler =StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape)    #(120, 4)
print(x_test.shape)     #(30, 4)

x_train = x_train.reshape(120, 2,2)
x_test = x_test.reshape(30, 2,2)

# #2. 모델구성
model = Sequential()
model.add(LSTM(units=350, input_shape=(2,2), 
               return_sequences=True))
model.add(LSTM(units=250))
model.add(Dense(100, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(3))
model.summary()


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',    # sparse_categorical_crossentropy 로 변경해도 가능
              metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min',patience=16, 
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
                      filepath = filepath + 'k48_07_' + date +'_'+ filename) 

model.fit(x_train, y_train, epochs=1, batch_size=3,
          validation_split=0.2,
          callbacks=[es,mcp],
          verbose=1)

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

print(y_test[:5])
y_predict = model.predict(x_test[:5])
print(y_predict)        # y_predict 결과값 모두 더하면 1

from sklearn.metrics import accuracy_score
import numpy as np

y_predict =  model.predict(x_test)
y_predict = np.argmax(y_predict, axis = 1)                  # 가장 큰 자릿값 뽑아냄   / axis=1 (가로축(행)), axis=0 (세로축(열))
print("y_pred(예측값) : ", y_predict)

y_test = np.argmax(y_test, axis=1)                          # 원핫을 안했으니까 필요없다.
print("y_test(원래값) : ", y_test)

acc = accuracy_score(y_test, y_predict)                     # 소수점 들어가는 실수 형태로 구성// error 발생
print(acc)

"""


"""







