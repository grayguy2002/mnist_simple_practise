import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,MaxPool2D,Dropout,Flatten,Dense
import os
from keras import backend as K
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)= mnist.load_data()
x_train , x_test = x_train/255.0 , x_test/255.0

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0],1,28,28)
    x_test = x_test.reshape(x_test.shape[0],1,28,28)
    input_shape = (1,28,28)
else:
    x_train = x_train.reshape(x_train.shape[0],28,28,1)
    x_test = x_test.reshape(x_test.shape[0],28,28,1)
    input_shape = (28,28,1)
    
    
    
model = tf.keras.models.Sequential([
     tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation = 'relu',input_shape=input_shape),
     MaxPool2D(pool_size=(2,2)),
     Dropout(rate=0.20),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(128 , activation = 'relu'),
     tf.keras.layers.Dense(10 , activation = 'softmax')])
 
model.compile(optimizer = 'adam',
               loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
               metrics = ['sparse_categorical_accuracy'])

model.fit(x_train,y_train,batch_size = 32, epochs = 5,validation_data = (x_test,y_test),
           validation_freq = 1)
model.summary()

model.save("models/mnist.h5",save_format='h5')




