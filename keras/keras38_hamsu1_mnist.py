import numpy as np
from tensorflow.keras.datasets import mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) #   (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) #   (10000, 28, 28) (10000,)

# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

x_train = x_train/255.
x_test = x_test/255.

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

print(np.unique(y_train, return_counts = True))     
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, Input

#2. 모델

inputs = Input(shape=(784,))
hidden1 = Dense(128, activation='linear') (inputs)
hidden2 = Dense(64) (hidden1)
hidden3 = Dense(32) (hidden2)
hidden4 = Dense(16) (hidden3)
hidden5 = Dense(8) (hidden4)
output = Dense(1) (hidden5)
model = Model(inputs=inputs, outputs=output)

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
                      filepath = filepath + 'k38_01_' + date +'_'+ filename)

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
