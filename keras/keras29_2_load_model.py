from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
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
    random_state=123, test_size=0.2)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print(np.min(x))
# print(np.max(x))
# print(dataset.feature_names)    
# print(dataset.DESCR)

#2. 모델구성(함수형 Model, input)

path = './_save/'
# path = '../save/'
# path = 'c:/study/_save/'
# model.save(path + 'keras29_1_save_model.h5')
model = load_model(path + 'keras29_1_save_model.h5')
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min',
                              patience=45, restore_best_weights=True,
                              verbose=1) 

model.fit(x_train, y_train, epochs=800, batch_size=32,
          callbacks=[earlyStopping],
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
R2 :  0.9473684210526315

"""


