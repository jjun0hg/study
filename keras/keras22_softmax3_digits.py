# 다중분류
import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) # (1797, 64)    (1797, )    // 64 = 8*8*1(흑백)
print(np.unique(y, return_counts=True))
#   (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),       // y에 들어갈 칼럼은 하나지만 o_h encoding 하면 칼럼(열) 10개로 늘어난다.
#   array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))          // 다중분류데이터(이미지 연산) // DNN방식
#   return_counts=True >> 각 숫자가 들어가는(나오는??) 횟수         // False 어떤 숫자가 나오는지

# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(datasets.images[4])
# plt.show()

y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,             
    random_state=333, 
    test_size=0.2,
    stratify=y 
)
print(y.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(64,)))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(10, activation='softmax'))

earlyStopping = EarlyStopping(monitor='val_loss', mode='min',
                              patience=30, restore_best_weights=True,
                              verbose=1) 
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',    # sparse_categorical_crossentropy 로 변경해도 가능
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=32,
          callbacks=[earlyStopping],
          validation_split=0.2,
          verbose=1)

#4. 평가, 검증
y_predict =  model.predict(x_test)
# print(y_predict)
y_predict = np.argmax(y_predict, axis = 1)                 
print("y_pred(예측값) : ", y_predict)

y_test = np.argmax(y_test, axis=1)
print("y_test(원래값) : ", y_test)

acc = accuracy_score(y_test, y_predict)
print(acc)


