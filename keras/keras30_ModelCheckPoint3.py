from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = './_save/'
# path = '../save/'
# path = 'c:/study/_save/'

#1. 데이터          /random_state = 123/ 1/ 365 /100000
datasets = load_boston()
x = datasets.data
y = datasets['target'] 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, 
    random_state=100000, test_size=0.2
)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성(함수형 Model, input)
# Input_dim=13(열,특성,feature), output = 1
input1  = Input(shape=(13,))
dense1 = Dense(50, activation = 'relu')(input1)
dense2 = Dense(40, activation = 'sigmoid')(dense1)
dense3 = Dense(30, activation = 'relu')(dense2)
dense4 = Dense(20, activation = 'linear')(dense3)
output1 = Dense(1, activation = 'linear')(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', mode='min',patience=1600, 
                  restore_best_weights=False,            # False 일때 MCP 데이터가 높아야한다.(이론상으로)
                   verbose=1) 

mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose = 1,
                      save_best_only=True,
                      filepath = path + 'MCP/keras30_ModelCheckPoint3.hdf5')

model.fit(x_train, y_train, epochs=80000, batch_size=8,
          callbacks=[es,mcp],verbose=1,
          validation_split=0.25)

model.save(path + "keras30_ModelCheckPoint3_save_model.h5")

#4. 평가, 예측
print("=====================1. 기본 출력 ========================")
mse, mae = model.evaluate(x_test, y_test)
print('mse : ', mse)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)      

print("=====================2. load_model 출력 ========================")
model2 = load_model(path + "keras30_ModelCheckPoint3_save_model.h5")
mse, mae = model2.evaluate(x_test, y_test)
print('mse : ', mse)

y_predict = model2.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)      

# 통상적으로 MCP 사용
print("=====================3. ModelCheckPoint 출력 ========================")
model3 = load_model(path + "MCP/keras30_ModelCheckPoint3.hdf5")
mse, mae = model3.evaluate(x_test, y_test)
print('mse : ', mse)

y_predict = model3.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)

