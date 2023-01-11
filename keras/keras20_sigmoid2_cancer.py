# 이진분류      // softmax
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.feature_names)

x = datasets['data']
y = datasets['target']
# print(x.shape, y.shape)     #   (569, 30) (569,)

y = pd.get_dummies(y, drop_first=False)
y = np.array(y)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,             
    random_state=333, 
    test_size=0.2,
    stratify=y 
)
    
#2. 모델구성
model = Sequential()
model.add(Dense(50, activation='linear', input_shape=(30,)))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))       # 0~1 사이의 값 출력

#3.컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',    # sparse_categorical_crossentropy 로 변경해도 가능
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=64,
          validation_split=0.2,
          verbose=1)
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=30,
                              restore_best_weights=True,
                              verbose=1) 

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=333, test_size=0.2
)
model.fit(x_train, y_train, epochs = 1000, batch_size=32,
          validation_split=0.2,
          callbacks = [earlyStopping],
          verbose = 1)


#4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print('loss, accuracy : ', loss)
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy :', accuracy)
y_predict =  model.predict(x_test)
# print(y_predict)
y_predict = np.argmax(y_predict, axis = 1)                 
print("y_pred(예측값) : ", y_predict)

y_test = np.argmax(y_test , axis=1)
print("y_test(원래값) : ", y_test)

acc = accuracy_score(y_test, y_predict)
print(acc)      #   0.9736842105263158


