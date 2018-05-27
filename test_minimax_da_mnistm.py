## Minimax version of domain-adpative NN
# Common filter for both mnist and mnistm
# 
# Some parts of the codes are from https://github.com/pumpikano/tf-dann.

import tensorflow as tf
import numpy as np
import cPickle as pkl
import matplotlib.pyplot as plt
import keras
#from keras.datasets import cifar10, cifar100
from keras.datasets import mnist
from utils import *


#########################################################################################################################

## Lower-level net (perturbation/filter)

def NN_filt(ins,scope='filt',reuse=False):
    with tf.variable_scope(scope,reuse=reuse):  
        W1 = tf.get_variable('W1',[5,5,3,32],initializer=tf.random_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('b1',[32],initializer=tf.constant_initializer(0.0))
        c1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(ins, W1, strides=[1,1,1,1], padding='SAME'),b1))
        p1 = tf.nn.max_pool(c1, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
        W2 = tf.get_variable('W2',[5,5,32,48],initializer=tf.random_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2',[48],initializer=tf.constant_initializer(0.0))
        c2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(p1, W2, strides=[1,1,1,1], padding='SAME'), b2))
        p2 = tf.nn.max_pool(c2, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
        out = tf.reshape(p2,[-1,7*7*48])
        reg = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
    return [out,reg]


## Upper-level net 

## Utility classifier
def NN_util(ins,scope='util',reuse=False):
    with tf.variable_scope(scope,reuse=reuse):
        W1 = tf.get_variable('W1',[7*7*48,nh],initializer=tf.random_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('b1',[nh],initializer=tf.constant_initializer(0.0))
        a1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(ins,W1),b1))
        W2 = tf.get_variable('W2',[nh,nh],initializer=tf.random_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2',[nh],initializer=tf.constant_initializer(0.0))
        a2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(a1,W2),b2))
        W3 = tf.get_variable('W3',[nh,Ku],initializer=tf.random_normal_initializer(stddev=0.1))
        b3 = tf.get_variable('b3',[Ku],initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(tf.matmul(a2,W3),b3)
        reg = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)
    return [out,reg]

## Privacy classifier
def NN_priv(ins,scope='priv',reuse=False):
    with tf.variable_scope(scope,reuse=reuse):
        W1 = tf.get_variable('W1',[7*7*48,nh],initializer=tf.random_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('b1',[nh],initializer=tf.constant_initializer(0.0))
        a1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(ins,W1),b1))
        W2 = tf.get_variable('W2',[nh,Kp],initializer=tf.random_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2',[Kp],initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(tf.matmul(a1,W2),b2)
        reg = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
    return [out,reg]


## minimax optimization
def minimax_kbeam(sess,feed_dict={}):
    # min step
    for it_min in range(min_step):
        fs = sess.run(loss,feed_dict)
        id_max = np.argmax(fs) 
        sess.run(optim_min[id_max],feed_dict)
    # max step
    for it_max in range(max_step):
        sess.run(optim_max,feed_dict)


def evaluate(X1test,X2test,y1test,y2test):
    batchsize = 1280
    n = X1test.shape[0]
    nbatch = np.int(np.ceil(np.float(n)/np.float(batchsize)))
    err_test_src = 0.
    err_test_tar = 0.
    for i in range(nbatch):
        ind = range(batchsize*i,min(batchsize*(1+i),n))
        err_test_src += sess.run(sumu,feed_dict={x:X1test[ind],yutil:y1test[ind], batch_size:batchsize}) 
        err_test_tar += sess.run(sumu,feed_dict={x:X2test[ind],yutil:y2test[ind], batch_size:batchsize})
    err_test_src /= np.float(n)
    err_test_tar /= np.float(n)

    return [err_test_src,err_test_tar]


#########################################################################################################################

result_dir = '/home/hammj/Dropbox/Research/AdversarialLearning/codes/results/icml18'


K = 2
max_step = 1
min_step = 1

ntrial = 10
max_iter = 10001
nskip = 100

Ku = 10
Kp = 2
nh = 100 
rho = 1E0 # Utility accuracy drops too much for rho smaller than 1 
lamb = 1E-12
batchsize = 128
lr = 1E-2

## Load data
(X1train, y1train), (X1test, y1test) = mnist.load_data()
X1train = np.tile(X1train.reshape((-1,28,28,1)),(1,1,1,3))
X1test = np.tile(X1test.reshape((-1,28,28,1)),(1,1,1,3))
y1train = keras.utils.to_categorical(y1train,10)
y1test = keras.utils.to_categorical(y1test,10)


## Load MNIST-M
mnistm = pkl.load(open('mnistm_data.pkl'))
X2train = mnistm['train']
X2test = mnistm['test']
X2valid = mnistm['valid']


## Create the model
x = tf.placeholder(tf.uint8, [None,28,28,3],'x') # Source domain
z = tf.placeholder(tf.uint8, [None,28,28,3],'z') # Target domain
yutil = tf.placeholder(tf.float32, [None, Ku],'yutil')
batch_size = tf.placeholder(tf.int32,[],'batch_size')

xfloat = tf.multiply(tf.cast(x, tf.float32),0.0039215686)
zfloat = tf.multiply(tf.cast(z, tf.float32),0.0039215686)


## Connect networks
filtx,reg_filt = NN_filt(xfloat)
filtz,_ = NN_filt(zfloat,reuse=True)
futilx,reg_util = NN_util(filtx)
futilz,_ = NN_util(filtz,reuse=True)

## y=0: real data,  y=1: fake data
yneg = tf.concat([tf.ones([batch_size,1]),tf.zeros([batch_size,1])],1)
ypos = tf.concat([tf.zeros([batch_size,1]),tf.ones([batch_size,1])],1)
loss_utilx = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=futilx,labels=yutil))
loss_utilz = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=futilz,labels=yutil))

vars_filt = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='filt')
vars_util = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='util')

fprivx = [[] for i in range(K)]
fprivz = [[] for i in range(K)]
loss_privx = [[] for i in range(K)]
loss_privz = [[] for i in range(K)]
loss = [[] for i in range(K)]
vars_priv = [[] for i in range(K)]
optim_max = [[] for i in range(K)]
optim_min = [[] for i in range(K)]

for i in range(K):
    fprivx[i],_ = NN_priv(filtx,'priv'+str(i),reuse=False)
    fprivz[i],_ = NN_priv(filtz,'priv'+str(i),reuse=True)
    loss_privx[i] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fprivx[i],labels=yneg))
    loss_privz[i] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fprivz[i],labels=ypos))
    loss[i] = rho*loss_utilx -0.5*loss_privx[i] -0.5*loss_privz[i] + lamb*(reg_filt+reg_util)
    vars_priv[i] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'priv'+str(i))
    optim_min[i] = tf.train.MomentumOptimizer(lr,0.9).minimize(loss[i],var_list=vars_filt+vars_util)
    optim_max[i] = tf.train.MomentumOptimizer(lr,0.9).minimize(-loss[i],var_list=vars_priv[i])


## test accuracy
iswrongu = tf.not_equal(tf.argmax(futilx,1), tf.argmax(yutil, 1))
sumu = tf.reduce_sum(tf.cast(iswrongu, tf.float32))

## Misc
saver = tf.train.Saver()


#####################################################################################################################

sess = tf.Session()
sess.run(tf.global_variables_initializer())

errs_src = np.nan*np.ones((int(np.ceil(max_iter/np.float(nskip))),ntrial))
errs_tar = np.nan*np.ones((int(np.ceil(max_iter/np.float(nskip))),ntrial))

gen_source_batch = batch_generator([X1train, y1train], batchsize)
gen_target_batch = batch_generator([X2train, y1train], batchsize)

print '\nK=%d, J=%d'%(K,max_step)

## Train
for trial in range(ntrial):
    print '%d/%d'%(trial,ntrial)
    cnt = 0
    sess.run(tf.global_variables_initializer())
    for it in range(max_iter):
        X, Y = gen_source_batch.next()
        Z, _ = gen_target_batch.next()
        feed_dict = {x:X, z:Z, yutil:Y, batch_size:batchsize}
        minimax_kbeam(sess, feed_dict)
        
        if it%nskip == 0:
            terr_src,terr_tar = evaluate(X1test,X2test,y1test,y1test)
            errs_src[cnt,trial] = terr_src
            errs_tar[cnt,trial] = terr_tar
            cnt += 1
            print 'step %d: test err src=%g, tar=%g'%(it,terr_src,terr_tar)


