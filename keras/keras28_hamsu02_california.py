# R2 0.55~0.6 이상
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, 
    random_state=123, test_size=0.2)


scaler = MinMaxScaler()
# scaler =StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# #2. 모델구성
# model = Sequential()
# model.add(Dense(10, input_shape=(8,), activation = 'relu'))
# model.add(Dense(20, activation ='relu'))
# model.add(Dense(30, activation ='relu'))
# model.add(Dense(40, activation ='relu'))
# model.add(Dense(20, activation ='relu'))
# model.add(Dense(10, activation ='relu'))
# model.add(Dense(10, activation ='relu'))
# model.add(Dense(1, activation = 'linear'))

#2. 모델구성(함수형 Model, input)
# Input_dim=13(열,특성,feature), output = 1
input1  = Input(shape=(8,))
dense1 = Dense(10, activation = 'relu')(input1)
dense2 = Dense(20, activation = 'relu')(dense1)
dense3 = Dense(30, activation = 'relu')(dense2)
dense4 = Dense(40, activation = 'relu')(dense3)
dense5 = Dense(20, activation = 'relu')(dense4)
dense6 = Dense(10, activation = 'relu')(dense5)
dense7 = Dense(10, activation = 'relu')(dense6)
output1 = Dense(1, activation = 'linear')(dense7)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min',
                              patience=45, restore_best_weights=True,
                              verbose=1) 
model.compile(loss='mae', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=350, batch_size=32,
              validation_split=0.25,
              callbacks=[earlyStopping],
              verbose=1)
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')          #list 형태는 그냥 넣어줘도됨
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
plt.grid()          
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('california')
plt.legend(loc='upper left')
plt.show()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)                           #   test 데이터로 평가
print('loss : ', loss)

y_predict = model.predict(x_test)                               #   최적의 가중치
from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))       
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)      



"""

loss :  0.33160167932510376
RMSE :  0.5208093280889106
R2 :   0.7729001654940569

"""



