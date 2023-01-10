import tensorflow as tf
print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')      # experimental 테스트중
# print(gpus)     [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

if(gpus):
    print("gpu on")
else:
    print("gpu X")

