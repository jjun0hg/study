import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([range(10)])   #   (10,)   (10,1) 
      
x = x.T
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
             [9,8,7,6,5,4,3,2,1,0]])
y = y.T
print(y.shape)      #   (10,)


#2. 모델구성
model = Sequential()
model.add(Dense(30, input_dim=1))        # input_dim = 열(컬럼, 피쳐, 특성)의 개수와 같다. 행무시 열우선 
model.add(Dense(40))
model.add(Dense(60))
model.add(Dense(90))
model.add(Dense(145))
model.add(Dense(180))
model.add(Dense(230))
model.add(Dense(180))
model.add(Dense(145))
model.add(Dense(90))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)      

result = model.predict([[9]])
print('[9]의 예측값 :', result)

"""


"""