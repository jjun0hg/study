from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
                                                    #   (60000, 5, 5, 1)
model.add(Conv2D(filters=10, kernel_size = (2,2),   #   filter = output(10)
                 input_shape=(5,5,1)))              #   (N(None), 4, 4, 10) // None == 데이터의 갯수
                                                    #   (batch_size(훈련의 갯수), rows, columns, chanels)
model.add(Conv2D(5, (2,2)))                         #   filter = output(5)(N, 3, 3, 5)
model.add(Flatten())                                #   (N, 45)
model.add(Dense(units = 10))                                #   (N, 10)
        # 인풋은 (batch_size, input_dim)
model.add(Dense(4, activation = 'relu')) # A, B, C, 렐루                                 #   (N, 1)    

model.summary()

###########################################################
# filter = 출력 공간의 차원
# kernel_size =   2D 컨볼루션 창의 높이와 너비를 지정하는 정수 또는 2개 정수의 튜플(목록)
#                 모든 공간 차원에 대해 동일한 값을 지정하는 단일 정수일 수 있다.
# strides = 높이와 너비에 따라 conv의 보폭을 지정하는 정수 또는 2개 정수의 튜플(목록)
# valid - same = 필터의 사이즈가 k이면 사방으로 k/2 만큼의 패딩을 준다
###########################################################


# save MCP 마지막 파일 제외하고 모두 삭제 bash 생성하기
