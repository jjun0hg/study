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
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
import time
model.compile(loss='mae', optimizer = 'adam')
start = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=1,
              validation_split=0.2,
              verbose=4)        # verbose =0 일때 속도가 더 빠르다. // verbose = 2 간략하게 보임(progress bar X) // verbose >=3 훈련 횟수만 보임
end = time.time()
print("걸린시간 : ", end - start)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)


