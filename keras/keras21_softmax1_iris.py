# 다중분류
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import pandas as pd
#1. 데이터
datasets = load_iris()
# print(datasets.DESCR)             # pandas .describe() // .info()
# print(datasets.feature_names)     # pandas.columns

x = datasets.data
y = datasets['target'] # datasets.target
# print(x.shape, y.shape)           # (150, 4) (150,)

# 인코딩

# num = np.unique(datasets['target'], axis=0)
# num = num.shape[0]
# encoding = np.eye(num)[datasets['target']]
# y = encoding
# print(y.shape)

y = pd.get_dummies(y, drop_first=False)
y = np.array(y)

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y.shape)        // (150,3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,             # False의 문제점 = 동일한 값만 나올수 있다. (항상 데이터 확인) 
    #random_state=333, 
    test_size=0.2,
    stratify=y                      # 분류형 데이터일 경우 가능하다
)
# print(y_train)
# print(y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(4,)))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='linear'))
model.add(Dense(3, activation='softmax'))       # 다중분류 = Dense(3, softmax) y의 종류(class) 갯수

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',    # sparse_categorical_crossentropy 로 변경해도 가능
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1, batch_size=3,
          validation_split=0.2,
          verbose=1)

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

print(y_test[:5])
y_predict = model.predict(x_test[:5])
print(y_predict)        # y_predict 결과값 모두 더하면 1

# from sklearn.metrics import accuracy_score
# import numpy as np

# y_predict =  model.predict(x_test)
# y_predict = np.argmax(y_predict, axis = 1)                  # 가장 큰 자릿값 뽑아냄   / axis=1 (가로축(행)), axis=0 (세로축(열))
# print("y_pred(예측값) : ", y_predict)

# y_test = np.argmax(y_test, axis=1)
# print("y_test(원래값) : ", y_test)

# acc = accuracy_score(y_test, y_predict)                     # 소수점 들어가는 실수 형태로 구성// error 발생
# print(acc)


