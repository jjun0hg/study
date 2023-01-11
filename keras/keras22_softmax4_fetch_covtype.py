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
datasets = fetch_covtype()      # sklearn 데이터 셋 안에 numpy로 저장되어 있다.
x = datasets.data               # numpy로 변환 되는 것이 아니다.
y = datasets['target']
print(type(x))
# print(x.shape, y.shape)     #   (581012, 54)   (581012,)
# print(np.unique(y, return_counts=True))     #   (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))
print(y.shape)

###################################################인코딩##############################################################
# get_dummies       // .values or numpy 로 변환 // index, header 자동생성 // numpy 자료형이 pandas를 바로 못받아들임
y = pd.get_dummies(y, drop_first=False)
# # y = y.values
# # y = y.to_numpy()
y = np.array(y)

# to_categorical
# y = to_categorical(y)
# print(type(y))
# # print(y[:10])
# # print(np.unique(y[:,0], return_counts=True))            # 모든 행의 0번째
# y = np.delete(y, 0, axis=1)

# OneHotEncoder     preprocessing = 전처리 // sparse=True default ==> Matrix 반환 // array가 필요하므로 False // y = y.toarray()
# Ohe = OneHotEncoder(sparse=True)
# # y = y.reshape(581012, 1)
# y = y.reshape(-1, 1)
# # # print(y.shape)
# # # Ohe.fit(y)                  
# # # print(y.shape)
# # y = Ohe.transform(y)            # 원하는 형식으로 변환 // 훈련시킨 결과(영향)에 대한 생성
# y = Ohe.fit_transform(y)
# y = y.toarray()
# print(y[:15])
# print(type(y))        scipy.sparse.~_maxtrix
# sparse = True .toarray()      // False 그대로

######################################################################################################################

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
                              patience=45, restore_best_weights=True,
                              verbose=1) 
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',    # sparse_categorical_crossentropy 로 변경해도 가능
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=64,
        callbacks=[earlyStopping],
        # validation_split=0.2,
        verbose=1)

# print(type(y))

#4. 평가, 검증
y_predict =  model.predict(x_test)
# print(y_predict)
y_predict = np.argmax(y_predict, axis = 1)                 
print("y_pred(예측값) : ", y_predict[:20])

y_test = np.argmax(y_test , axis=1)
print("y_test(원래값) : ", y_test[:20])

acc = accuracy_score(y_test, y_predict)
print(acc)      #   0.8708725248057279

