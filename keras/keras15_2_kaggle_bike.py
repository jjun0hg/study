import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path = './_data/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)    
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

print(train_csv.shape)    
                            
print(test_csv.shape)
print(train_csv.describe()) 

###### 결측치 처리  1. 제거#####
print(train_csv.isnull().sum())        
train_csv = train_csv.dropna()         
print(train_csv.isnull().sum())         

x = train_csv.drop(['count','casual','registered'], axis=1)   # axis=축, x값만 남기기
print(x.shape)    
y = train_csv['count']
print(y.shape)
#print(y)
#print(y.shape)  

x_train, x_test, y_train, y_test = train_test_split(x, y,
                        train_size=0.9, shuffle=True, random_state=519)
print(x_train.shape, x_test.shape)  
print(y_train.shape, y_test.shape)  

#2. 모델구성
model = Sequential()
# model.add(Dense(36, input_dim=9))
model.add(Dense(15, input_dim=8, activation='relu'))   # default
model.add(Dense(30, activation = 'relu'))
model.add(Dense(15, activation = 'relu'))
model.add(Dense(14, activation = 'relu'))
model.add(Dense(13, activation = 'relu'))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(11, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(9, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

#3. 컴파일, 훈련
#loss = mae or mse optimizer= 'adam', matrix[mae or mse]
import time
model.compile(loss='mse', optimizer='adam',
                metrics=['mae'])
start = time.time()
model.fit(x_train, y_train, epochs=800, batch_size=32)
end = time.time()
print("걸린시간 : ", end - start)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
# print(y_predict)

# 결측치 처리 x

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)  # RMSE :  0.5567980375745774
                        
# 제출
y_submit = model.predict(test_csv)
print(y_submit)
# print(y_submit.shape)   #(715, 1)

# .to_csv()를 사용
# submission_0105.csv
# print(submission)
submission['count'] = y_submit  # y_submit 저장
# print(submission)
submission.to_csv(path +"submission_01061035.csv")


"""
model.add(Dense(32, input_dim=8, activation='relu'))   # default
model.add(Dense(64, activation = 'sigmoid'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'sigmoid'))
model.add(Dense(32, activation = 'sigmoid'))
model.add(Dense(32, activation = 'sigmoid'))
model.add(Dense(32, activation = 'sigmoid'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))
RMSE :  151.43057139621573

model.compile(loss='mae', optimizer='adam',
                metrics=['mse'])
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=32


"""

