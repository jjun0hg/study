from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape) #   (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   #   (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts = True))   

#2. 모델
model = Sequential()
model.add(Conv2D(512, (3,3), input_shape=(32,32,3),     
                padding="same",  
                use_bias=True,            
                kernel_initializer="random_normal",
                activation='relu'))                      
model.add(MaxPooling2D()) 
model.add(Conv2D(256, (3,3), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Flatten())                                           # 40000
model.add(Dense(32, activation = 'relu'))                      # input_shape(40000,) //(60000,40000)// (batch_size, input_dim)
model.add(Dense(100, activation = 'softmax'))                  

#3. 컴파일, 훈련
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam',
              metrics = ['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min',patience=50, 
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
                      filepath = filepath + 'k35_04_' + date +'_'+ filename)
model.fit(x_train, y_train, epochs=1000, 
            verbose= 1, 
            batch_size=250,
            validation_split=0.2,
            callbacks=[es,mcp])

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', result[1])

