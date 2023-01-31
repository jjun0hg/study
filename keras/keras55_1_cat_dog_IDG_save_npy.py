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
    'D:\_data/train/',
    target_size=(250,250),     # 이미지 크기 조정
    batch_size=20,              # pytorch는 batch로 분리후 집어넣는다.
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
    # Found 160 images belonging to 2 classes.
)

xy_test = test_datagen.flow_from_directory(
    'D:\_data/test1/',
    target_size=(250,250),      # 이미지 크기 조정
    batch_size=20,              # pytorch는 batch로 분리후 집어넣는다.
    class_mode='binary',
    color_mode='r',
    shuffle=True
    # Found 120 images belonging to 2 classes.
)

print(xy_train[0][0].shape)     # (10000, 250, 250, 1)
print(xy_train[0][1].shape)     # (10000,)

print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
print(type(xy_train[0][1])) #<class 'numpy.ndarray'>

np.save('C:\_data/train/x_train.npy', arr=xy_train[0][0])
np.save('C:\_data/train/y_train.npy', arr=xy_train[0][1])


np.save('C:\_data/test1/x_test.npy', arr=xy_test[0][0])
np.save('C:\_data/test1/y_test.npy', arr=xy_test[0][1])


