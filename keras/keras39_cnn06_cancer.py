# 이진분류      // softmax
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.feature_names)
x = datasets['data']
y = datasets['target']
# print(x.shape, y.shape)     #   (569, 30) (569,)

y = pd.get_dummies(y, drop_first=False)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, 
    random_state=333, test_size=0.2, stratify=y )

scaler = MinMaxScaler()
# scaler =StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape)    #(455, 30)
print(x_test.shape)     #(114, 30)

x_train = x_train.reshape(455, 30, 1, 1)
x_test = x_test.reshape(114, 30, 1, 1)

print(x_train.shape)    #(455, 30)
print(x_test.shape)     #(114, 30)

# #2. 모델구성
model = Sequential()
model.add(Conv2D(64, (1,1), input_shape=(30, 1, 1)))
model.add(Dense(32, activation='linear'))    
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='linear'))    


#3.컴파일, 훈련
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
                      filepath = filepath + 'k39_06_' + date +'_'+ filename) 
 
model.fit(x_train, y_train, epochs=10000, batch_size=32,
          validation_split=0.2,
          callbacks=[es,mcp],
          verbose=1)

#4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print('loss, accuracy : ', loss)
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy :', accuracy)
y_predict =  model.predict(x_test)
# print(y_predict)
y_predict = np.argmax(y_predict, axis = 1)                 
print("y_pred(예측값) : ", y_predict)

y_test = np.argmax(y_test , axis=1)
print("y_test(원래값) : ", y_test)

acc = accuracy_score(y_test, y_predict)
print(acc)      
#   


