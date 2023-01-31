import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

xy_train = train_datagen.flow_from_directory(
    'C:\_data/horse-or-human/horse-or-human',
    target_size=(250,250),     # 이미지 크기 조정
    batch_size=20,              # pytorch는 batch로 분리후 집어넣는다.
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
    
)

xy_test = test_datagen.flow_from_directory(
    'C:\_data/horse-or-human/horse-or-human',
    target_size=(250,250),      # 이미지 크기 조정
    batch_size=20,              # pytorch는 batch로 분리후 집어넣는다.
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

horse = train_datagen.flow_from_directory(
    'C:\_data/horse/',
    target_size=(250,250),     # 이미지 크기 조정
    # batch_size=20,              # pytorch는 batch로 분리후 집어넣는다.
    class_mode=None,
    # color_mode='rgb',
    shuffle=True
)

human = train_datagen.flow_from_directory(
    'C:\_data/human/',
    target_size=(250,250),     # 이미지 크기 조정
    # batch_size=20,              # pytorch는 batch로 분리후 집어넣는다.
    class_mode=None,
    # color_mode='rgb',
    shuffle=True
)

dog = train_datagen.flow_from_directory(
    'C:\_data/dog/',
    target_size=(250,250),     # 이미지 크기 조정
    # batch_size=20,              # pytorch는 batch로 분리후 집어넣는다.
    class_mode=None,
    # color_mode='rgb',
    shuffle=True
)

cat = train_datagen.flow_from_directory(
    'C:\_data/cat/',
    target_size=(250,250),     # 이미지 크기 조정
    # batch_size=20,              # pytorch는 batch로 분리후 집어넣는다.
    class_mode=None,
    # color_mode='rgb',
    shuffle=True
    # Found 160 images belonging to 2 classes.
)


print(horse[0][0].shape)     
print(human[0][0].shape)
# np.save('C:\_data/horse-or-human/x_train.npy', arr=xy_train[0][0])
# np.save('C:\_data/horse-or-human/y_train.npy', arr=xy_train[0][1])
# np.save('C:\_data/horse-or-human/x_test.npy', arr=xy_test[0][0])
np.save('C:\_data/horse-or-human/y_test.npy', arr=xy_test[0][1])

x_train = np.load('C:\_data/horse-or-human/x_train.npy')     #, arr=xy_train[0][0])
y_train = np.load('C:\_data/horse-or-human/y_train.npy')     #, arr=xy_train[0][1])
x_test = np.load('C:\_data/horse-or-human/x_test.npy')       #, arr=xy_test[0][0]
y_test = np.load('C:\_data/horse-or-human/y_test.npy')       #, arr=xy_test[0][0]

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

model = Sequential()
model.add(Conv2D(256, (3,3), input_shape=(250, 250, 3)))
model.add(MaxPooling2D((3, 3)))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(MaxPooling2D((3, 3)))
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(MaxPooling2D((3, 3)))
model.add(Conv2D(16, (3,3), activation='linear'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))           # 0과 1  // softmax쓰려면 2가 되어야함
model.add(Dense(1, activation='sigmoid'))           # 0과 1  // softmax쓰려면 2가 되어야함
# model.add(Dense(2, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
#               metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', mode='min',patience=20, 
                  restore_best_weights=True,                
                   verbose=1)

hist = model.fit(x_train, y_train,
                    batch_size=16, 
                    # steps_per_epoch=16,
                    epochs=100,
                    validation_data= (x_test,y_test),
                    callbacks=[es])
                    # validation_steps=4)


accuracy = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])              #훈련의 마지막 값
print('val_loss : ', val_loss[-1])      #훈련의 마지막 값
print('accuracy : ', accuracy[-1])      #훈련의 마지막 값
print('val_acc : ', val_acc[-1])        #훈련의 마지막 값

result = model.predict(horse)
print('result : ', result)