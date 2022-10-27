import tensorflow as tf
from tensorflow.keras import Model
from keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
import pandas as pd
from keras import backend as K
import matplotlib
import matplotlib.pyplot as plt
#find the prediction error cases in the test dataset
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

model = load_model('F:/Eclipse-workspace/tensorflow_practice/models/mnist.h5')

pred = []

pred = model.predict(x_test)
pred = np.array(pred)
pred_index = np.argmax(pred,axis = 1)
print(pred_index)
judge = y_test - pred_index
print(judge)
a = 0
index_error = []
for i in range(len(judge)):
    if  judge[i] != 0:
        index_error.append(i)
        continue
    else:
        a +=1
        continue
#print(a)#a = 14
#print(len(judge))
#print(index_error)
y_test_pred_error = {}
y_test_predict = {}
final_dic = {}
for i in range(len(index_error)):
    y_test_pred_error[i] = y_test[index_error[i]]#真实值
    y_test_predict[i] = pred_index[index_error[i]] #预测失败值
    final_dic[i] = {y_test_pred_error[i]:y_test_predict[i]}#得到最后模型的预测失败例子共len(index_error)个；(i:{真实值：预测值})
arr_x = pd.Series(y_train)
arr_y = pd.Series(y_test)
arr_x = arr_x.value_counts()
arr_y = arr_y.value_counts()
arr_x.sort_index(inplace=True)
arr_y.sort_index(inplace=True)

plt.pie(arr_x,labels=['0','1','2','3','4','5','6','7','8','9'])
plt.title('training dataset')
plt.show()
plt.pie(arr_y,labels=['0','1','2','3','4','5','6','7','8','9'])
plt.title('validation dataset')
plt.show()

