## Generative model of mnist digits.

import time
import numpy as np
import scipy.io
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from mpl_toolkits.axes_grid1 import ImageGrid
#from __future__ import print_function
import keras
#from keras.datasets import cifar10, cifar100
from keras.datasets import mnist
from utils import *
import sys

##########################################################################################################################

def decoder(ins,scope,reuse=False,d=10): # z -> x
    batch_size = tf.shape(ins)[0]
    with tf.variable_scope(scope,reuse=reuse):  
        W1 = tf.get_variable('W1',[d,7*7*64],initializer=tf.random_normal_initializer(stddev=0.01))
        b1 = tf.get_variable('b1',[7*7*64],initializer=tf.constant_initializer(0.0))
        a1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(ins,W1),b1))
        W2 = tf.get_variable('W2',[5,5,32,64],initializer=tf.random_normal_initializer(stddev=0.01))
        b2 = tf.get_variable('b2',[32],initializer=tf.constant_initializer(0.0))
        a2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d_transpose(tf.reshape(a1,[-1,7,7,64]),W2,[batch_size,14,14,32],strides=[1,2,2,1],padding='SAME'),b2))
        W3 = tf.get_variable('W3',[5,5,1,32],initializer=tf.random_normal_initializer(stddev=0.01))
        b3 = tf.get_variable('b3',[1],initializer=tf.constant_initializer(0.0))
        a3 = tf.nn.sigmoid(tf.nn.bias_add(tf.nn.conv2d_transpose(a2,W3,[batch_size,28,28,1],strides=[1,2,2,1],padding='SAME'),b3))
        out = a3#tf.reshape(a3,[-1,28,28,1])    
        reg = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)
    return [out,reg]

def discriminator(ins,scope,reuse=False,nh=50):
    with tf.variable_scope(scope,reuse=reuse):
        W1 = tf.get_variable('W1',[5,5,1,16],initializer=tf.random_normal_initializer(stddev=0.01))
        b1 = tf.get_variable('b1',[16],initializer=tf.constant_initializer(0.0))
        c1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(ins, W1, strides=[1,1,1,1], padding='SAME'),b1))
        p1 = tf.nn.max_pool(c1, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
        W2 = tf.get_variable('W2',[5,5,16,32],initializer=tf.random_normal_initializer(stddev=0.01))
        b2 = tf.get_variable('b2',[32],initializer=tf.constant_initializer(0.0))
        c2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(p1, W2, strides=[1,1,1,1], padding='SAME'), b2))
        p2 = tf.nn.max_pool(c2, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
        a2 = tf.reshape(p2,[-1,7*7*32])
        W3 = tf.get_variable('W3',[7*7*32,K],initializer=tf.random_normal_initializer(stddev=0.01))
        b3 = tf.get_variable('b3',[K],initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(tf.matmul(a2,W3),b3)
        reg = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)
    return [out,reg]


def minimax_kbeam(sess, feed_dict):
    # min-step
    for it_min in range(min_step):
        fs = sess.run(loss,feed_dict)
        id_max = np.argmax(fs) 
        sess.run(optim_min[id_max],feed_dict)
    
    # max-step
    for it_max in range(max_step):
        #fs = sess.run(loss,feed_dict)
        #id_max = np.argmax(fs) 
        #sess.run(optim_max[id_max],feed_dict)
        sess.run(optim_max,feed_dict)



###########################################################################################################################################

D = 28*28*1
d = 10
K = 5
max_step = 1
min_step = 1
batchsize = 128
lr_gen = 1E-3
lr_disc = 1E-3
niter = 10001


## Load data
(X1train, y1train), (X1test, y1test) = mnist.load_data()
X1train = X1train.reshape((-1,28,28,1)).astype('float32')/255.
X1test = X1test.reshape((-1,28,28,1)).astype('float32')/255.
y1train = keras.utils.to_categorical(y1train,10)
y1test = keras.utils.to_categorical(y1test,10)

## Create the model
x1 = tf.placeholder(tf.float32, [None, 28,28,1])
x2 = tf.placeholder(tf.float32, [None, d])

## Generator
dec,reg_dec = decoder(x2,'mnist_decoder',d=d)
vars_dec = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='mnist_decoder')

## Discriminators
disc_real=[[] for i in range(K)]
disc_fake=[[] for i in range(K)]
loss=[[] for i in range(K)]
vars_disc=[[] for i in range(K)]
optim_max=[[] for i in range(K)]
optim_min=[[] for i in range(K)]
reg_disc=[[] for i in range(K)]


for i in range(K):
    disc_real[i], reg_disc[i]= discriminator(x1,'disc'+str(i),False)
    disc_fake[i],_ = discriminator(dec,'disc'+str(i),True)
    
    loss[i] = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real[i], labels=tf.ones_like(disc_real[i]))) \
    -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake[i], labels=tf.zeros_like(disc_fake[i])))
    '''    
    eps = 1E-12
    ## original loss - doesn't work
    loss[i] = tf.reduce_mean(tf.log(tf.maximum(eps,tf.minimum(tf.nn.sigmoid(disc_real[i]),1.-eps)))) \
        +tf.reduce_mean(tf.log(tf.subtract(1.,tf.maximum(eps,tf.minimum(tf.nn.sigmoid(disc_fake[i]),1-eps)))))
    '''
    vars_disc[i] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'disc'+str(i))
    optim_min[i] = tf.train.AdamOptimizer(lr_gen).minimize(loss[i],var_list=vars_dec)
    optim_max[i] = tf.train.AdamOptimizer(lr_disc).minimize(-loss[i],var_list=vars_disc[i])


## Misc
#saver = tf.train.Saver()

#########################################################################################################

## Init
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#sess.run(tf.initialize_all_variables())

## Load pre-trained autoencoders
#saver_dec.restore(sess, dir_model+'/mnist_decoder.ckpt')

if True:
    X2test = np.random.uniform(-1.,1.,size=((64,d)))
    #np.save(dir_result+'/X2test.npy',X2test)
else:
    pass
    #X2test = np.load(dir_result+'/X2test.npy')

gen_batch = batch_generator([X1train, y1train], batchsize)


## Burn-in period for v, when using u from autoencoder
for it in range(0):
    X1, Y1 = gen_batch.next()
    X2 = np.random.uniform(-1.,1.,size=(batchsize,d))    
    feed_dict = {x1:X1, x2:X2}
    sess.run(optim_max,feed_dict)
    if it%100 == 0:
        fs = sess.run(loss,feed_dict)
        id_max = np.argmax(fs) 
        f = np.max(fs)
        print 'step %d: f=%g, id_max=%d'%(it,f,id_max)


## Minimax 
for it in range(niter):
    X1, Y1 = gen_batch.next()
    X2 = np.random.uniform(-1.,1.,size=(batchsize,d))    
    feed_dict = {x1:X1, x2:X2}
    minimax_kbeam(sess, feed_dict)
    if it%100 == 0:
        fs = sess.run(loss,feed_dict)
        id_max = np.argmax(fs) 
        f = np.max(fs)
        print 'step %d: f=%g, id_max=%d'%(it,f,id_max)
        
    if it%5000 == 0:
        X2test_ = sess.run(dec,feed_dict={x2:X2test})
        imshow_grid(X2test_[:64,:].reshape((64,28,28)),[8,8])
        plt.suptitle('t=%d'%(it),size=12)
        #plt.show(block=False)    
        plt.pause(3.)
        
        #with matplotlib.backends.backend_pdf.PdfPages(dir_result+'/gan_mnist_K%d_max%d_%d.pdf'%(K,max_step,it)) as pdf:
        #    pdf.savefig(bbox_inches='tight')


raw_input('Press any key to continue')


