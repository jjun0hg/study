from tensorflow.keras.datasets import cifar10
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape) #   (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   #   (10000, 32, 32, 3) (10000, 1)

model = Sequential()
model.add(Conv2D(256, (3,3), input_shape=(32,32,3),         #(32, 32, 3)
                padding="same",  
                activation='relu'))                            #(27, 27, 128) 
model.add(MaxPooling2D())        
model.add(Conv2D(128, (3,3), activation='relu'))               #(26, 26, 64)
model.add(Conv2D(64, (3,3), activation='relu'))                                #(25, 25, 64)
model.add(Flatten())                                            # 40000
model.add(Dense(32, activation = 'relu'))                      # input_shape(40000,) //(60000,40000)// (batch_size, input_dim)
model.add(Dense(10, activation = 'softmax'))

#2. 모델
model = Sequential()
model.add(Conv2D(256, (3,3), input_shape=(32,32,3),         #(32, 32, 3)
                padding="same",  
                strides= 2,
                activation='relu'))                            #(27, 27, 128) 
model.add(MaxPooling2D()) 
model.add(Conv2D(128, (3,3), activation='relu'))               #(26, 26, 64)
model.add(Conv2D(64, (3,3), activation='relu'))                                #(25, 25, 64)
model.add(Flatten())                                            # 40000
model.add(Dense(32, activation = 'relu'))                      # input_shape(40000,) //(60000,40000)// (batch_size, input_dim)
model.add(Dense(10, activation = 'softmax'))                
# model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam',
              metrics = ['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min',patience=20, 
                  restore_best_weights=True,              
                   verbose=1)
import datetime                                             # 데이터 형식으로
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose = 1,
                      save_best_only=True,
                      filepath = filepath + 'k35_03_' + date +'_'+ filename)

model.fit(x_train, y_train, epochs=300, 
            verbose= 1, 
            batch_size=250,
            validation_split=0.2,
            callbacks=[es,mcp])

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', result[1])

