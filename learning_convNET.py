import numpy as np
import keras
from PIL import Image
import os
import sys
from keras.layers import Conv2D
from keras.models import Sequential
import scipy.misc
import tensorflow as tf
from keras import optimizers


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

if (len(sys.argv)>3):
  out_folder=sys.argv[1]
  w0=float(sys.argv[2])
  w1=float(sys.argv[3])
else:
  out_folder='data/cnn_confidence/'
  w0=1/.05
  w1=1

train_folder = 'data/small_training/'
train_groundtruth_folder='data/small_training_groundtruth/'
test_folder='data/small_testing/'

def images_from_folder(folder):
  images=[]
  for filename in os.listdir(folder):
    if filename[0]!='.':
      img= Image.open(folder+filename)
      images.append(np.array(img))
  return images

def weighted_loss(weights):
  def class_weighted_pixelwise_crossentropy(target,output):
    output=tf.clip_by_value(output,10e-8,1.-10e-8)
    return -tf.reduce_mean(target*weights[0]*tf.log(output)+(1-target)*weights[1]*tf.log(1-output))
  return class_weighted_pixelwise_crossentropy

def IoU_metric(y_true,y_pred):
  y_true=tf.reshape(y_true,[-1])
  y_pred=tf.reshape(y_pred,[-1])
  y_pred=tf.cast((y_pred>=.5),tf.float32)
  TP=tf.reduce_sum(y_true*y_pred)
  TN=tf.reduce_sum((1-y_true)*(1-y_pred))
  FP=tf.reduce_sum((y_true)*(1-y_pred))
  FN=tf.reduce_sum((1-y_true)*(y_pred))
  return TP/(TP+FP+FN)


def IoU_numpy(y_true,y_pred):
  y_true=y_true.reshape(-1)
  y_pred=y_pred.reshape(-1)
  TP=np.sum(y_true*y_pred)
  TN=np.sum((1-y_true)*(1-y_pred))
  FP=np.sum((y_true)*(1-y_pred))
  FN=np.sum((1-y_true)*(y_pred))
  return TP/(TP+FP+FN)

train=np.array(images_from_folder(train_folder))
train_groundtruth=np.array(images_from_folder(train_groundtruth_folder))
test=np.array(images_from_folder(test_folder))


train=train/255-.5
test=test/255-.5
train_groundtruth=train_groundtruth>128

train=train.reshape(train.shape+(1,))
test=test.reshape(test.shape+(1,))

train_groundtruth=train_groundtruth.reshape(train_groundtruth.shape+(1,)).astype(np.float32)


model= Sequential()
model.add(Conv2D(40,input_shape=(train.shape)[1:],kernel_size=(3,3),dilation_rate=(1,1),activation='relu',padding='same'))
model.add(Conv2D(80,kernel_size=(3,3),activation='relu',dilation_rate=(1,1),padding='same'))
model.add(Conv2D(80,kernel_size=(3,3),activation='relu',dilation_rate=(1,1),padding='same'))
model.add(Conv2D(80,kernel_size=(3,3),activation='relu',dilation_rate=(2,2),padding='same'))
model.add(Conv2D(80,kernel_size=(3,3),activation='relu',dilation_rate=(4,4),padding='same'))
model.add(Conv2D(80,kernel_size=(3,3),activation='relu',dilation_rate=(8,8),padding='same'))
model.add(Conv2D(1,kernel_size=(1,1),activation='sigmoid',dilation_rate=(1,1),padding='same'))

model.compile(loss=weighted_loss([w0,w1]),optimizer='adam',metrics=[IoU_metric,'accuracy'])
model.fit(train,train_groundtruth,epochs=100,validation_split=.1,batch_size=5)

result=model.predict(test)
result=result.reshape(result.shape[:-1])
for i in range(len(test)):
  scipy.misc.imsave(out_folder+'test_{0}.jpg'.format(i),result[i])


