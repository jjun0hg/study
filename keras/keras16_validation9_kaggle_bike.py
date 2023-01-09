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
# print(x_train.shape, x_test.shape)  
# print(y_train.shape, y_test.shape)  
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234, test_size=0.2)

#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=8, activation='relu'))   
model.add(Dense(16, activation ='relu'))
model.add(Dense(10, activation ='relu'))
model.add(Dense(1, activation = 'linear'))

#3. 컴파일, 훈련
#loss = mae or mse optimizer= 'adam', matrix[mae or mse]
import time
start = time.time()
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=16,
          validation_split=0.2)
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


RMSE :  156.32053757797001

"""

