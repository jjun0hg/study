# 다중분류
import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

#1. 데이터
datasets = fetch_covtype()
x = datasets.data              # 데이터 뽑아올때 numpy로 변환되는건지??
y = datasets['target']
# print(x.shape, y.shape)     #   (581012, 54)   (581012,)
# print(np.unique(y, return_counts=True))     #   (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))
print(y.shape)

###################################################인코딩##############################################################
# get_dummies       // .values or numpy 로 변환
y = pd.get_dummies(y, drop_first=False)
y = np.array(y)

# to_categorical
# y = to_categorical(y)
# y = np.delete(y, 0, axis=1)


# OneHotEncoder     // sparse=True default ==> Matrix 반환 // array가 필요하므로 False // .toarray()
# Ohe = OneHotEncoder(sparse=False)
# y = y.reshape(-1, 1)
# print(y.shape)
# Ohe.fit(y)                  
# print(y.shape)
# y = Ohe.transform(y)            # 원하는 형식으로 변환

# sparse = True .toarray()      // False 그대로

#########################################################################################################

# num = np.unique(datasets['target'], axis=0)
# num = num.shape[0]
# encoding = np.eye(num)[datasets['target']]
# y = encoding
# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,             
    random_state=333, 
    test_size=0.2,
    stratify=y 
)

#2. 모델구성
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(54,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(7, activation='softmax'))       # 열은 하나인데 칼럼 7개

earlyStopping = EarlyStopping(monitor='val_loss', mode='min',
                              patience=30, restore_best_weights=True,
                              verbose=1) 
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',    # sparse_categorical_crossentropy 로 변경해도 가능
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100000, batch_size=64,
          callbacks=[earlyStopping],
          validation_split=0.2,
          verbose=1)
              
# print(type(y))
#4. 평가, 검증
y_predict =  model.predict(x_test)
# print(y_predict)
y_predict = np.argmax(y_predict, axis = 1)                 
print("y_pred(예측값) : ", y_predict)

y_test = np.argmax(y_test , axis=1)
print("y_test(원래값) : ", y_test)

acc = accuracy_score(y_test, y_predict)
print(acc)

