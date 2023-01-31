import numpy as np

# np.save('./_data/brain/brain_x_train.npy', arr=xy_train[0][0])
# np.save('./_data/brain/brain_y_train.npy', arr=xy_train[0][1])

# np.save('./_data/brain/brain_x_test.npy', arr=xy_test[0][0])
# np.save('./_data/brain/brain_y_test.npy', arr=xy_test[0][1])

x_train = np.load('./_data/brain/brain_x_train.npy')    
y_train = np.load('./_data/brain/brain_y_train.npy')

x_test = np.load('./_data/brain/brain_x_test.npy')
y_test = np.load('./_data/brain/brain_y_test.npy')

print(x_train.shape, x_test.shape)      #(160, 200, 200, 1) (10, 200, 200, 1)
print(y_train.shape, y_test.shape)      #(160,) (10,)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(160, (2,2), input_shape=(200, 200, 1)))
model.add(Conv2D(90, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))           # 0과 1  // softmax쓰려면 2가 되어야함

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])

hist = model.fit(x_train, y_train,
                    batch_size=16, 
                    # steps_per_epoch=16,
                    epochs=100,
                    validation_data= (x_test, y_test))
                    # validation_steps=4)

accuracy = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])              #훈련의 마지막 값
print('val_loss : ', val_loss[-1])      #훈련의 마지막 값
print('accuracy : ', accuracy[-1])      #훈련의 마지막 값
print('val_acc : ', val_acc[-1])        #훈련의 마지막 값

# reduceLR(MCP,ES 쪽에 있음)
