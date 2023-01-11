from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target'] 

# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# print(np.min(x))
# print(np.max(x))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, 
    random_state=333, test_size=0.2)

scaler = MinMaxScaler()
# scaler =StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
x_test = scaler.fit_transform(x_test)

# print(np.min(x))
# print(np.max(x))
# print(dataset.feature_names)    
# print(dataset.DESCR)

#2. 모델구성
# Input_dim=13(열,특성,feature), output = 1
model = Sequential()
model.add(Dense(26, input_dim=13, activation = 'linear'))
model.add(Dense(40, activation = 'linear'))
model.add(Dense(30, activation = 'linear'))
model.add(Dense(20, activation = 'linear'))
model.add(Dense(1, activation = 'linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=800, batch_size=32,
          validation_split=0.25)

#4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


"""

개선 전  R2 :  0.7276881435330604
개선 후 (minmax)  R2 :  0.7575389404015904
개선 후 (standard scaler)  R2 :  0.7637876692969494
R2 :  0.7490650147812239


"""


