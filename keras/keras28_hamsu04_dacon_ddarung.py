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
path = './_data/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)    
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

###### 결측치 처리  1. 제거#####
print(train_csv.isnull().sum())         
train_csv = train_csv.dropna()          
print(train_csv.isnull().sum())         
print(submission.shape)                 

x = train_csv.drop(['count'], axis=1)   # axis=축
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, 
    random_state=2134, test_size=0.2)

scaler = MinMaxScaler()
# scaler =StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# #2. 모델구성
# model = Sequential()
# model.add(Dense(27,input_shape=(9,)))
# model.add(Dense(10, activation ='relu'))
# model.add(Dense(5, activation ='relu'))
# model.add(Dense(1, activation = 'linear'))

input1  = Input(shape=(9,))
dense1 = Dense(27, activation = 'relu')(input1)
dense2 = Dense(10, activation = 'relu')(dense1)
dense3 = Dense(5, activation = 'relu')(dense2)
output1 = Dense(1, activation = 'linear')(dense3)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
import matplotlib.pyplot as plt

model.compile(loss='mae', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min',
                              patience=300, restore_best_weights=True,
                              verbose=3) 
hist = model.fit(x_train, y_train, epochs=100000, batch_size=32,
                callbacks=[earlyStopping],
                validation_split=0.5)

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')          #list 형태는 그냥 넣어줘도됨
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
plt.grid()          
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('dacon ddarung')
plt.legend(loc='upper left')
plt.show()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)  

# 제출
y_submit = model.predict(test_csv)
submission['count'] = y_submit  # y_submit 저장
submission.to_csv(path +"submission_01050251.csv")


"""


loss :  30.237001419067383
RMSE :  44.74754924146947


"""

