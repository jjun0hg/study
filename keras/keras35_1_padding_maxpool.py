import numpy as np
from tensorflow.keras.datasets import mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) #   (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) #   (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

print(np.unique(y_train, return_counts = True))     
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

#2. 모델
model = Sequential()
model.add(Conv2D(128, (3,3), input_shape=(28,28,1),         #(28, 28, 128)
                padding='same',     # valid      
                strides= 2,
                activation='relu'))                        
model.add(MaxPooling2D())                                   #(14, 14, 128)
model.add(Conv2D(64, (2,2), padding='same'))                #(28, 28, 64)
model.add(Conv2D(64, (2,2)))                                #(27, 27, 64)
model.add(Flatten())                                        # 46656
model.add(Dense(32, activation = 'relu'))                   
model.add(Dense(10, activation = 'softmax'))                

model.summary()

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
                #      filepath = path + 'MCP/keras35_ModelCheckPoint.hdf5')
                      filepath = filepath + 'k35_01_' + date +'_'+ filename)

model.fit(x_train, y_train, epochs=3000, 
            verbose= 1, 
            batch_size=32,
            validation_split=0.2,
            callbacks=[es,mcp])

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', result[1])

# loss :  0.10004928708076477
# acc :  0.9708999991416931

# padding 적용시...
#   acc :  0.9608949349914169

#   MaxPooling2D 적용시
#   acc :  0.9797999858856201
