from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras import layers
import numpy as np
#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape) #   (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   #   (10000, 32, 32, 3) (10000, 1)
x_train = x_train / 255
x_test = x_test / 255

print(np.unique(y_train, return_counts = True))     
#       (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 
#        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 
#        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
#        68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
#        85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]), array([500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500], dtype=int64))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

#2. 모델
model = Sequential()
model.add(Conv2D(512, (2,2), input_shape=(32,32,3),     
                padding="same",  
                use_bias=True,            
                kernel_initializer="random_normal",
                activation='relu'))                      
model.add(layers.MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3,3), activation='relu'))               #(26, 26, 64)
model.add(layers.MaxPooling2D((2, 2)))              
model.add(Conv2D(128, (3,3), activation='relu'))                                #(25, 25, 64)
model.add(layers.MaxPooling2D((2, 2)))              
model.add(Conv2D(64, (2,2), activation='relu'))                                #(25, 25, 64)
model.add(Flatten())                                           # 40000
model.add(Dense(32, activation = 'relu'))                      # input_shape(40000,) //(60000,40000)// (batch_size, input_dim)
model.add(Dense(100, activation = 'softmax'))                  

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
                #      filepath = path + 'MCP/keras30_ModelCheckPoint3.hdf5')
                      filepath = filepath + 'k34_03_' + date +'_'+ filename)
model.fit(x_train, y_train, epochs=250, 
            verbose= 1, 
            batch_size=250,
            validation_split=0.2,
            callbacks=[es,mcp])

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', result[1])

# loss :  2.556501865386963
# acc :  0.353300005197525