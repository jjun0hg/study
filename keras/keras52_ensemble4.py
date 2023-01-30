import numpy as np
#1. 데이터

x_datasets = np.array([range(100), range(301, 401)]).transpose()
print(x_datasets.shape)        #(100, 2)   # 삼성전자 시가, 고가

y1 = np.array(range(2001, 2101)) # (100,)    # 삼성전자의 하루뒤 종가
y2 = np.array(range(201, 301)) # (100,)    # 아모레의 하루뒤 종가

from sklearn.model_selection import train_test_split        # default 0.75
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x_datasets, y1, y2, train_size=0.7, random_state=1234
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델1.
input1 = Input(shape=(2,))
dense1 = Dense(100, activation='relu', name='ds11')(input1)
dense2 = Dense(50, activation='relu', name='ds12')(dense1)
dense3 = Dense(30, activation='relu', name='ds13')(dense2)
dense4 = Dense(20, activation='relu', name='ds14')(dense3)
output1 = Dense(10, activation='linear', name='ds15')(dense4)


#2-4. 모델병합
from tensorflow.keras.layers import concatenate,Concatenate

merge1 = concatenate([output1], name='mg1')
merge2 = Dense(50, activation='relu', name='mg2')(merge1)
merge3 = Dense(30, name='mg3')(merge2)
merge3 = Dense(20, name='mg3')(merge2)
merge4 = Dense(5, name='mg4')(merge3)
last_output = Dense(1, name='last')(merge4)

#2-5. 모델5 분기1
dense5 = Dense(200, activation='relu', name='ds41')(last_output)
dense5 = Dense(150, activation='relu', name='ds42')(dense5)
dense5 = Dense(30, activation='relu', name='ds43')(dense5)
dense5 = Dense(20, activation='relu', name='ds44')(dense5)
output5 = Dense(10, activation='linear', name='ds45')(dense5)

#2-5. 모델5 분기2
dense6 = Dense(200, activation='relu', name='ds51')(last_output)
dense6 = Dense(150, activation='relu', name='ds52')(dense6)
dense6 = Dense(30, activation='relu', name='ds53')(dense6)
dense6 = Dense(20, activation='relu', name='ds54')(dense6)
output6 = Dense(10, activation='linear', name='ds55')(dense6)

model = Model([input1],[output5, output6])
model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer= 'adam')
model.fit([x1_train], [y1_train, y2_train], epochs=1000, batch_size=8)

#4. 평가,예측
loss = model.evaluate([x1_test], [y1_test, y2_test])
print('loss = ', loss)

