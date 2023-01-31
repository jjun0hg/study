import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#1. 데이터

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
    './_data/brain/train/',
    target_size=(250,250),     # 이미지 크기 조정
    batch_size=10,              # pytorch는 batch로 분리후 집어넣는다.
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
    # Found 160 images belonging to 2 classes.
)

# x = (160,150,150,1) // 150,150 = 사진크기
# y = (160,)

xy_test = train_datagen.flow_from_directory(
    './_data/brain/test/',
    target_size=(250,250),      # 이미지 크기 조정
    batch_size=10,              # pytorch는 batch로 분리후 집어넣는다.
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
    # Found 120 images belonging to 2 classes.
)

# print(xy_train[0])
# print(xy_train[0][0])
# print(xy_train[0][1]) 
print(xy_train[0][0].shape)     # (6, 200, 200, 1) // batch_size에 따라 달라짐
print(xy_train[0][1].shape) 

# <keras.preprocessing.image.DirectoryIterator object at 0x00000249F2375670>
# print(xy_test)
# <keras.preprocessing.image.DirectoryIterator object at 0x000001FCA4484EE0>

print(type(xy_train))       #<class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))    #<class 'tuple'>
print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
print(type(xy_train[0][1])) #<class 'numpy.ndarray'>

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(160, (2,2), input_shape=(250, 250, 1)))
model.add(Conv2D(90, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))           # 0과 1  // softmax쓰려면 2가 되어야함

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])

hist = model.fit(   xy_train[0][0], xy_train[0][1],
                    #xy_train,
                    batch_size=16, 
                    # steps_per_epoch=16,
                    epochs=100,
                    validation_data= (xy_test[0][0], xy_test[0][1]),
                    validation_split=0.2)
                    # validation_steps=4)
                    
accuracy = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])              #훈련의 마지막 값
print('val_loss : ', val_loss[-1])      #훈련의 마지막 값
print('accuracy : ', accuracy[-1])      #훈련의 마지막 값
print('val_acc : ', val_acc[-1])        #훈련의 마지막 값

# plot
import matplotlib.pyplot as plt 
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')          #list 형태는 그냥 넣어줘도됨
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
plt.plot(hist.history['val_acc'], c='black', marker='.', label='val_acc')
plt.grid() 
plt.title('fit_generator')
plt.legend(loc='upper left')
plt.show()


