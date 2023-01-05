import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path = './_data/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# train_csv = pd.read_csv('./_data/ddarung/train.csv', index_col=0)    # 원래 해야하는거, index_col=0 == 0번째는 데이터 아니다.
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv)    #(1459, 10) , count는 y값이므로 제외해야한다. input_dim=9

print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')
print(train_csv.info())     #Non-Null Count 결측치(1459- 1457 =2), (1459-1457 = 2), (1459-1450=9) ...
                            # 결측치가 있는 데이터는 삭제해버린다.
print(test_csv.info())
print(train_csv.describe()) #std = 표준편차, 50% = 중간값

###### 결측치 처리  1. 제거#####
print(train_csv.isnull().sum())         # null값 모두 더하기
train_csv = train_csv.dropna()          # 결측치 제거
print(train_csv.isnull().sum())         # null값 모두 더하기
print(train_csv.shape)                  # (1328, 10)

x = train_csv.drop(['count'], axis=1)   # axis=축
print(x)    #   [1459 rows x 9 columns]
y = train_csv['count']
print(y)
print(y.shape)  # (1459, )

x_train, x_test, y_train, y_test = train_test_split(x, y,
                        train_size=0.7, shuffle=True, random_state=1234)
print(x_train.shape, x_test.shape)  #   (929, 9) (399, 9)
print(y_train.shape, y_test.shape)  #   (929,) (399,)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=9))
model.add(Dense(9))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(10))
model.add(Dense(6))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(4))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(10))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
#loss = mae or mse optimizer= 'adam', matrix[mae or mse]
model.compile(loss='mse', optimizer='adam',
                metrics=['mae'])
model.fit(x_train, y_train, epochs=100000, batch_size=32)

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print(y_predict)

# 결측치 처리 x

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)  # RMSE :  83.02001881026747


# 제출
y_submit = model.predict(test_csv)






