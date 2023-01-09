from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (506, 13)     (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=333, test_size=0.2
)

#2. 모델구성
model = Sequential()
# model.add(Dense(5, input_dim=13))
model.add(Dense(5, input_shape=(13,)))     # input_shape
model.add(Dense(40000))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
import time
model.compile(loss='mae', optimizer = 'adam')
hist = model.fit(x_train, y_train, epochs=300, batch_size=1,     # hist = history
              validation_split=0.2,
              verbose=1)        #   verbose =0 일때 속도가 더 빠르다. // verbose = 2 간략하게 보임(progress bar X) // verbose >=3 훈련 횟수만 보임
print("===================================")              
print(hist)                     #   <keras.callbacks.History object at 0x0000011656E1B610>     
print("===================================")              
print(hist.history)             #   key, value == dictionary,  value = list형태
print("===================================")              
print(hist.history['val_loss'])             #   key, value == dictionary,  value = list형태

import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')          #list 형태는 그냥 넣어줘도됨
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
plt.grid()          
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('boston loss')
plt.legend(loc='upper left')
plt.show()



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)


