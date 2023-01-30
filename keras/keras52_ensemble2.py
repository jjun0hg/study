import numpy as np
#1. 데이터

x1_datasets = np.array([range(100), range(301, 401)]).transpose()
print(x1_datasets.shape)        #(100, 2)   # 삼성전자 시가, 고가
x2_datasets = np.array([range(101,201), range(411, 511), range(150,250)]).T
print(x2_datasets.shape)        #(100, 3)   # 아모레 시가, 고가, 종가
x3_datasets = np.array([range(100,200), range(1301, 1401)])
x3_datasets = x3_datasets.reshape(100,2)
print(x3_datasets.shape)

y = np.array(range(2001, 2101)) # (100,)    # 삼성전자의 하루뒤 종가

from sklearn.model_selection import train_test_split        # default 0.75
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test , y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y, train_size=0.7, random_state=1234
)

print(x1_train.shape, x2_train.shape, x3_train.shape, y_train.shape)
print(x2_test.shape, x2_test.shape, x3_test.shape, y_test.shape)

#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델1.
input1 = Input(shape=(2,))
dense1 = Dense(100, activation='relu', name='ds11')(input1)
dense2 = Dense(50, activation='relu', name='ds12')(dense1)
dense3 = Dense(30, activation='relu', name='ds13')(dense2)
dense4 = Dense(20, activation='relu', name='ds14')(dense3)
output1 = Dense(10, activation='linear', name='ds15')(dense4)

#2-2. 모델2.
input2 = Input(shape=(3,))
dense21 = Dense(150, activation='relu', name='ds21')(input2)
dense22 = Dense(50, activation='relu', name='ds22')(dense21)
dense23 = Dense(30, activation='relu', name='ds23')(dense22)
dense24 = Dense(20, activation='relu', name='ds24')(dense23)
output2 = Dense(10, activation='linear', name='ds25')(dense24)

#2-2. 모델3.
input3 = Input(shape=(2,))
dense31 = Dense(200, activation='relu', name='ds31')(input3)
dense32 = Dense(150, activation='relu', name='ds32')(dense31)
dense33 = Dense(30, activation='relu', name='ds33')(dense32)
dense34 = Dense(20, activation='relu', name='ds34')(dense33)
output3 = Dense(10, activation='linear', name='ds35')(dense34)

#2-3. 모델병합
from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2, output3], name='mg1')
merge2 = Dense(50, activation='relu', name='mg2')(merge1)
merge3 = Dense(30, name='mg3')(merge2)
merge3 = Dense(20, name='mg3')(merge2)
merge4 = Dense(5, name='mg4')(merge3)
last_output = Dense(1, name='last')(merge4)

model = Model(inputs=[input1, input2, input3], outputs = last_output)

model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer= 'adam')
model.fit([x1_train, x2_train, x3_train], y_train, epochs=1000, batch_size=8)

#4. 평가,예측
loss = model.evaluate([x1_test, x2_test, x3_test], y_test)

print(loss)

