import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
# train_set = pd.read_csv('./_data/ddarung/train.csv', index_col=0)    # 원래 해야하는거, index_col=0 == 0번째는 데이터 아니다.
test_set = pd.read_csv(path + 'test.csv', index_col=0)
submitssion = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_set)    #(1459, 10) , count는 y값이므로 제외해야한다. input_dim=9

print(train_set.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')
print(train_set.info())     #Non-Null Count 결측치(1459- 1457 =2), (1459-1457 = 2), (1459-1450=9) ...
                            # 결측치가 있는 데이터는 삭제해버린다.
print(test_set.info())
print(train_set.describe()) #std = 표준편차, 50% = 중간값

x = train_set.drop(['count'], axis=1)   # axis=축
print(x)    #[1459 rows x 9 columns]
y = train_set['count']
print(y)
print(y.shape)  # (1459, )
x_train, x_test, y_train, y_test = train_test_split(x, y,
                        train_size=0.7, shuffle=True, random_state=115)
#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=9))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
#loss = mae or mse optimizer= 'adam', matrix[mae or mse]
model.compile(loss='mse', optimizer='adam',
                metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=32)

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))






