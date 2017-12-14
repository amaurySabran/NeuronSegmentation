import numpy as np
from PIL import Image
import os
import sys
import scipy.misc
from sklearn.mixture import GaussianMixture

#We learn two GMMS, one for the background and one for the foreground
#We output the confidence of a pixel being foreground c= p(fore)/(p(fore)+p(back))
#These confidence maps are written in separate folders to use them in C++

#In order to learn the GMM, we can feed it only the texture (pixel intensity) or intensity+gradient value

out_folder='data/test_gmm_confidence/'
train_folder = 'data/small_training/'
train_groundtruth_folder='data/small_training_groundtruth/'
test_folder='data/small_testing/'
test_groundtruth_folder='data/small_testing_groundtruth/'

WRITE_RESULTS=True
USE_GRADIENT_FOR_GMM=False

def images_from_folder(folder):
  images=[]
  files=os.listdir(folder)
  files=[f for f in files if f[0]!='.']
  f=files[0]
  prefix="_".join(f.split('_')[:-1])
  extension=f.split('.')[-1]
  for i in range(len(files)):
    filename=prefix+'_'+str(i)+'.'+extension
    img= Image.open(folder+filename)
    images.append(np.array(img))
  return images

def images_to_folder(im_array,folder):
  for i in range(len(im_array)): 
    scipy.misc.imsave(folder+'confidence_{0}.jpg'.format(i), im_array[i])

def IoU_score(y_true,y_pred):
  y_true=y_true.reshape(-1)
  y_pred=y_pred.reshape(-1)
  TP=np.sum(y_true*y_pred)
  TN=np.sum((1-y_true)*(1-y_pred))
  FP=np.sum((y_true)*(1-y_pred))
  FN=np.sum((1-y_true)*(y_pred))
  return TP/(TP+FP+FN)

def add_gradient(X):
  g=np.linalg.norm(np.array(np.gradient(X)),axis=0)
  g=g.reshape(g.shape+(1,))
  X=X.reshape(X.shape+(1,))
  return np.concatenate([X,g],-1)

train=np.array(images_from_folder(train_folder))/255
train_groundtruth=np.array(images_from_folder(train_groundtruth_folder))>128
test=np.array(images_from_folder(test_folder))/255
test_groundtruth=np.array(images_from_folder(test_groundtruth_folder))>128
test_shape=test.shape

if USE_GRADIENT_FOR_GMM:
  train_f=add_gradient(train)
  test_f=add_gradient(test)
else:
  train_f=train.reshape(train.shape+(1,))
  test_f=test.reshape(test.shape+(1,))

gmm1=GaussianMixture(6,verbose=2)
gmm2=GaussianMixture(6,verbose=2)
gmm1.fit(train_f[train_groundtruth==0].reshape((-1,train_f.shape[-1])))
gmm2.fit(test_f[train_groundtruth==1].reshape((-1,train_f.shape[-1])))

print("gmm negative")
print("mean: {0}, std dev: {1},weights :{2}".format(gmm1.means_,np.sqrt(gmm1.covariances_),gmm1.weights_))
print("gmm positive")
print("mean: {0}, std dev: {1},weights :{2}".format(gmm2.means_,np.sqrt(gmm2.covariances_),gmm2.weights_))

test_score1=np.exp(gmm1.score_samples(test_f.reshape((-1,test_f.shape[-1]))))
test_score2=np.exp(gmm2.score_samples(test_f.reshape((-1,test_f.shape[-1]))))

confidence=test_score2/(test_score1+test_score2)# à affiner?
confidence.reshape(test_shape)

print("IoU: {0}".format(IoU_score(test_groundtruth,confidence>.5)))

## write confidences
if WRITE_RESULTS:
  images_to_folder(confidence.reshape(test_shape),out_folder)



