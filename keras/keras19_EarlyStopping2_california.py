# R2 0.55~0.6 이상
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234, test_size=0.2)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_shape=(8,), activation = 'relu'))
model.add(Dense(20, activation ='relu'))
model.add(Dense(30, activation ='relu'))
model.add(Dense(40, activation ='relu'))
model.add(Dense(20, activation ='relu'))
model.add(Dense(10, activation ='relu'))
model.add(Dense(10, activation ='relu'))
model.add(Dense(1, activation = 'linear'))

#3. 컴파일, 훈련
import matplotlib.pyplot as plt

model.compile(loss='mae', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min',
                              patience=10, restore_best_weights=True,
                              verbose=1) 

hist = model.fit(x_train, y_train, epochs=500, batch_size=32,
              validation_split=0.25,
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
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234, test_size=0.2)
model.add(Dense(10, input_shape=(8,), activation = 'relu'))
model.add(Dense(20, activation ='relu'))
model.add(Dense(30, activation ='relu'))
model.add(Dense(40, activation ='relu'))
model.add(Dense(20, activation ='relu'))
model.add(Dense(10, activation ='relu'))
model.add(Dense(10, activation ='relu'))
model.add(Dense(1, activation = 'linear'))
hist = model.fit(x_train, y_train, epochs=500, batch_size=32,
                    validation_split=0.25,
                    verbose=1)
earlyStopping = EarlyStopping(monitor='val_loss', mode='min',
                              patience=10, restore_best_weights=True,
                              verbose=1) 
loss :  0.5092134475708008
RMSE :  0.7064828922973457
R2 :  0.6310516321982463
"""



