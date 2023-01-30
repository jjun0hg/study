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
    './_data/brain/train/',
    target_size=(200,200),     # 이미지 크기 조정
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
    target_size=(200,200),      # 이미지 크기 조정
    batch_size=10,              # pytorch는 batch로 분리후 집어넣는다.
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
    # Found 120 images belonging to 2 classes.
)

# print(xy_train[0])
# print(xy_train[0][0])
# print(xy_train[0][1]) 
print(xy_train[0][0].shape)     # (6, 200, 200, 1)
print(xy_train[0][1].shape) 

# <keras.preprocessing.image.DirectoryIterator object at 0x00000249F2375670>
# print(xy_test)
# <keras.preprocessing.image.DirectoryIterator object at 0x000001FCA4484EE0>

print(type(xy_train))       #<class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))    #<class 'tuple'>
print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
print(type(xy_train[0][1])) #<class 'numpy.ndarray'>







