import numpy as np

x_train = np.load('C:\_data/train/x_train.npy')     #, arr=xy_train[0][0])
y_train = np.load('C:\_data/train/y_train.npy')     #, arr=xy_train[0][1])
x_test = np.load('C:\_data/test1/x_test.npy')       #, arr=xy_test[0][0]
y_test = np.load('C:\_data/test1/y_test.npy')       #, arr=xy_test[0][1])   

print(x_train.shape,y_test.shape)
print(y_train.shape,y_test.shape)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(200, (2,2), input_shape=(250, 250, 3)))
model.add(Conv2D(90, (3,3), activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))           # 0과 1  // softmax쓰려면 2가 되어야함
# model.add(Dense(2, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
#               metrics=['acc'])

hist = model.fit(x_train, y_train,
                    batch_size=20, 
                    # steps_per_epoch=16,
                    epochs=100,
                    validation_data= (x_test,y_test))
                    # validation_steps=4)

accuracy = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])              #훈련의 마지막 값
print('val_loss : ', val_loss[-1])      #훈련의 마지막 값
print('accuracy : ', accuracy[-1])      #훈련의 마지막 값
print('val_acc : ', val_acc[-1])        #훈련의 마지막 값