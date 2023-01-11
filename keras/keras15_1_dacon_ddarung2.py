# 결측치 처리 O
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path = './_data/ddarung/'
# path = '../_data/ddarung/'
# path = 'c:/study/_data/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)    # train_csv = pd.read_csv('./_data/ddarung/train.csv', index_col=0)==원래 해야하는거// index_col=0 == 0번째는 데이터 아니다.
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

# print(train_csv)    #(1459, 10) , count는 y값이므로 제외해야한다. input_dim=9
# print(train_csv.columns)
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
print(submission.shape)                 # (715, 1) //평가 데이터에도 결측치가 존재한다(삭제로는 해결 x)



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
model.add(Dense(19,input_dim=9))
model.add(Dense(15))
model.add(Dense(17))
model.add(Dense(19))
model.add(Dense(15))
model.add(Dense(13))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(1))

#3. 컴파일, 훈련
#loss = mae or mse optimizer= 'adam', matrix[mae or mse]
import time
model.compile(loss='mae', optimizer='adam',
                metrics=['mse'])
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=7)
end = time.time()
print("걸린시간 : ", end - start)
# cpu 걸린시간 :  51.54655790328979560
# gpu 걸린시간 :  22.688528060913086

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
# print(y_predict)

# 결측치 처리 x

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)  # RMSE :  83.02001881026747
                        # RMSE :  53.88971756086701
                        
# submission.to_csv(path +"submission_0105.csv", mode='w')

# 제출
y_submit = model.predict(test_csv)
# print(y_submit)
# print(y_submit.shape)   #(715, 1)

# .to_csv()를 사용
# submission_0105.csv
# print(submission)
submission['count'] = y_submit  # y_submit 저장
# print(submission)
submission.to_csv(path +"submission_01050251.csv")


"""
model = Sequential()
model.add(Dense(90, input_dim=9))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(1)) 
RMSE :  53.569496808788806



x_train, x_test, y_train, y_test = train_test_split(x, y,
                        train_size=0.8, shuffle=True, random_state=1234)
model = Sequential()
# model.add(Dense(36, input_dim=9))
model.add(Dense(9,input_dim=9))
model.add(Dense(9))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(15))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))
model.fit(x_train, y_train, epochs=100, batch_size=32)

"""

