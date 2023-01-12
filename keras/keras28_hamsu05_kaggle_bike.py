import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path = './_data/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)    
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)



###### 결측치 처리  1. 제거#####
      
train_csv = train_csv.dropna()         
     
x = train_csv.drop(['count','casual','registered'], axis=1)   # axis=축, x값만 남기기
print(x.shape)    
y = train_csv['count']
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, 
    random_state=123, test_size=0.1)


scaler = MinMaxScaler()
# scaler =StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# #2. 모델구성
# model = Sequential()
# model.add(Dense(40, input_shape=(8,), activation='relu'))   
# model.add(Dense(20, activation ='relu'))
# model.add(Dense(15, activation ='relu'))
# model.add(Dense(10, activation ='relu'))
# model.add(Dense(2, activation ='relu'))
# model.add(Dense(1, activation = 'linear'))

#2. 모델구성
input1  = Input(shape=(8,))
dense1 = Dense(40, activation = 'relu')(input1)
dense2 = Dense(20, activation = 'relu')(dense1)
dense3 = Dense(15, activation = 'relu')(dense2)
dense4 = Dense(10, activation = 'linear')(dense3)
dense5 = Dense(2, activation = 'linear')(dense4)
output1 = Dense(1, activation = 'linear')(dense5)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
import matplotlib.pyplot as plt

model.compile(loss='mae', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min',
                              patience=3000, restore_best_weights=True,
                              verbose=1) 
hist = model.fit(x_train, y_train, epochs=100000, batch_size=32,
                callbacks=[earlyStopping],
                validation_split=0.25)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)
# 결측치 처리 x

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)  
                        
# 제출
y_submit = model.predict(test_csv)
print(y_submit)

submission['count'] = y_submit  # y_submit 저장
submission.to_csv(path +"submission_01061035.csv")


"""




"""

