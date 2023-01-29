import pandas as pd
import numpy as np
#1. 데이터
path = './_data/stock/'
samsung = pd.read_csv(path + '삼성전자 주가.csv',  encoding='cp949', index_col=0, thousands=',')
amore = pd.read_csv(path + '아모레퍼시픽 주가.csv', encoding='cp949',  index_col=0, thousands=',')
# samsung = (samsung - np.mean(samsung, axis=0)) / np.std(samsung, axis=0)
# amore = (amore - np.mean(amore, axis=0)) / np.std(amore, axis=0)

# print(samsung.isnull().sum())
samsung =  samsung.dropna()
amore = amore.dropna()
# print(samsung.shape, amore.shape) (1977, 17)(2220, 17)

x_1 = samsung.loc[:, ['시가','고가','저가','종가','거래량']]
y_1 = amore.loc[:, ['시가','고가','저가','종가','거래량']]
# print(x_1,y_1)

x = np.array(x_1).T
y = np.array(y_1).T
samsung_ = x[::-1]
amore_ = y[::-1]

# print(samsung_.shape,amore_.shape)  # (5, 1977) (5, 2210)

timesteps = 5  # y는 없다.
def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

x_predict = split_x(samsung_, timesteps)
x_predict = x_predict.reshape(5,1977)
print(x_predict.shape)


from sklearn.model_selection import train_test_split        # default 0.75
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    samsung_, amore_, x_predict, train_size=0.7, random_state=1234
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델1.
input1 = Input(shape=(1977,))
dense1 = Dense(100, activation='relu', name='ds11')(input1)
dense2 = Dense(50, activation='relu', name='ds12')(dense1)
dense3 = Dense(30, activation='relu', name='ds13')(dense2)
dense4 = Dense(20, activation='relu', name='ds14')(dense3)
output1 = Dense(10, activation='relu', name='ds15')(dense4)


#2-2. 모델2.
input2 = Input(shape=(2210,))
dense21 = Dense(100, activation='linear', name='ds21')(input2)
dense22 = Dense(50, activation='linear', name='ds22')(dense21)
dense23 = Dense(30, activation='linear', name='ds23')(dense22)
dense24 = Dense(20, activation='linear', name='ds24')(dense23)
output2 = Dense(10, activation='linear', name='ds25')(dense24)

#2-3. 모델병합
from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2], name='mg1')
merge2 = Dense(15, activation='relu', name='mg2')(merge1)
merge3 = Dense(12, name='mg3')(merge2)
merge4 = Dense(12, name='mg4')(merge3)
last_output = Dense(1, name='last')(merge4)

model = Model(inputs=[input1, input2], outputs = last_output)

model.summary()

model.compile(loss = 'mse', optimizer= 'adam')
model.fit([x1_train, x2_train], y_train, epochs=1000, batch_size=16)

#4. 평가,예측
loss = model.evaluate([x1_test, x2_test], y_test)
result = model.predict([x1_test, x2_test])
print( result )
print(loss)

