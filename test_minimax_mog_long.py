## test_minimax_mog_long.py
## Jihun Hamm, 2017

## Parts of codes are from https://github.com/poolio/unrolled_gan

import os
import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from collections import OrderedDict
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from scipy.stats import entropy
import sys

ds = tf.contrib.distributions
slim = tf.contrib.slim

######################################################################################################

def sample_mog(n_mixture=8, std=0.01, radius=1.0):
    thetas = np.linspace(0, 2 * np.pi, n_mixture)
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
    cat = ds.Categorical(tf.zeros(n_mixture))
    comps = [ds.MultivariateNormalDiag([xi, yi], [std, std]) for xi, yi in zip(xs.ravel(), ys.ravel())]
    mog = ds.Mixture(cat, comps)
    return mog
    

def generator(z, output_dim=2, n_hidden=128, n_layer=2):
    with tf.variable_scope("generator"):
        h = slim.stack(z, slim.fully_connected, [n_hidden] * n_layer, activation_fn=tf.nn.tanh)
        x = slim.fully_connected(h, output_dim, activation_fn=None)
    return x

def discriminator(x, scope='discriminator', n_hidden=128, n_layer=2, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        h = slim.stack(x, slim.fully_connected, [n_hidden] * n_layer, activation_fn=tf.nn.tanh)
        log_d = slim.fully_connected(h, 1, activation_fn=None)
    return log_d


## minimax optimization
def minimax2(sess,feed_dict={}):
    # min step
    for it_min in range(params['min_step']):
        fs = sess.run(loss,feed_dict)
        id_max = np.argmax(fs) 
        sess.run(optim_min[id_max],feed_dict)
    # max step
    for it_max in range(params['max_step']):
        sess.run(optim_max,feed_dict)


## Epsilon-steepest descent
solvers.options['maxiters'] = 20
solvers.options['show_progress'] = False
def EpsSteepestDescentDirection(grads):
    # solve min_{g \in L_eps(u)} \|g\|^2
    # min {a} a'ZZ'a, s.t. a>=0 and a'1 = 1.
    n = len(grads)
    Z = np.zeros((0,))
    for i in range(n):
        L = len(grads[i])
        for l in range(L):
            Z = np.concatenate((Z,grads[i][l].flatten()))
    #for i in range(len(grads[0])):
    #    print grads[0][i].shape
    d = len(Z)/n #=49666
    Z = Z.reshape(n,d)
    P = matrix(2.*Z.dot(Z.T))
    q = matrix(0.0, (n,1))
    G = matrix(-np.eye(n))
    h = matrix(0.0, (n,1))
    A = matrix(1.0, (1,n))
    b = matrix(1.0)
    # 1/2x'Px + q'x, Gx<=h, Ax=b
    sol = solvers.qp(P,q,G,h,A,b)
    z = np.dot(sol['x'].T,Z)
    if False:#np.random.rand(1)>.95:
        print n
        print sol['x']
        print sol['status']
        #if any(np.dot(Z,z.T)<0):
        #    print np.dot(Z,z.T)
        #    print 'negative angle!!!!'
    return [sol['x'],z]


def minimax3(sess, eps=1E-6, feed_dict={}):
    gnsq = 1E12
    # min step
    for it_min in range(params['min_step']):
        fs = sess.run(loss,feed_dict)
        fmax = np.max(fs)
        # Find v's with min gradient norm
        ids_eps = np.where(fs>=fmax-eps)[0]
        if len(ids_eps)==1:
            sess.run(optim_min[ids_eps[0]],feed_dict)
            gnsq = sess.run(gradnormsq[ids_eps[0]],feed_dict)            
        else:
            if False: # Heuristic -- choose the min norm direction
                id_min = np.argmin(gnsqs)
                sess.run(optim_min[ids_eps[id_min]],feed_dict)
            else: # Solve QP
                #grads  = sess.run(grad[ids_eps],feed_dict)
                #print ids_eps
                grads = []#np.zeros((n))
                for i,ids in enumerate(ids_eps):
                    grads.append(sess.run(grad[ids],feed_dict))
                a = np.zeros((params['K']))
                a[ids_eps], z = EpsSteepestDescentDirection(grads)
                gnsq = (z**2).sum()
                #print np.sqrt(gnsq)
                #feed_dict['al'] = al
                sess.run(optim_min_weighted,feed_dict={al:a})
                # Suppose the z \in L(u) is z=\sum_i a_i nabla fi
                # Define the weighted loss phi_a = \sum_i a_i fi and min over u.
    # max step
    for it_max in range(params['max_step']):
        sess.run(optim_max,feed_dict)
        
    return gnsq


def compute_p_orig():
    nrep = 5000 # x128
    nbin = 20
    samples_orig = np.zeros((0,2))
    for i in range(nrep):
        samples_orig = np.concatenate((samples_orig,sess.run(data)),0)
    p_orig = np.histogram2d(samples_orig[:,0], samples_orig[:,1], bins=nbin, range=[[-2.,2.],[-2.,2.]])[0]
    p_orig += 1.
    p_orig = p_orig.reshape((nbin**2))
    p_orig = p_orig/p_orig.sum()
    return p_orig


def compute_jsd(p_orig):
    nrep = 500
    nbin = 20
    samples_gan = np.zeros((0,2))
    for i in range(nrep):
         samples_gan = np.concatenate((samples_gan,sess.run(samples)),0)
    p_gan = np.histogram2d(samples_gan[:,0], samples_gan[:,1], bins=nbin, range=[[-2.,2.],[-2.,2.]])[0]
    p_gan += 1.
    p_gan = p_gan.reshape((nbin**2))
    p_gan = p_gan/p_gan.sum()
    p_avg = (p_orig + p_gan)/2.0
    jsd = 0.5*entropy(p_orig,p_avg) + 0.5*entropy(p_gan,p_avg)
    
    return jsd



#####################################################################################################
## Params
    
params = dict(
    batch_size=128,
    disc_learning_rate=1e-4,
    gen_learning_rate=1e-3, 
    beta1=0.5,
    epsilon=1e-8,
    z_dim=256,
    x_dim=2,
    gamma=0,
    K=2, # 1,2,5,10
    max_step=1, 
    min_step=1,
    ntrial=10,
    max_iter=50001,
    viz_every=5000,
    nskip=1000,
)


## Construct model and training ops
tf.reset_default_graph()

mog = sample_mog()
data = mog.sample(params['batch_size'])
noise = ds.Normal(tf.zeros(params['z_dim']), 
                  tf.ones(params['z_dim'])).sample(params['batch_size'])
# Construct generator and discriminator nets
with slim.arg_scope([slim.fully_connected], weights_initializer=tf.orthogonal_initializer(gain=1.4)):
    samples = generator(noise, output_dim=params['x_dim'])
    real_score = discriminator(data)
    fake_score = discriminator(samples, reuse=True)

gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")


## Parallel discriminators
disc_real = [[] for i in range(params['K'])]
disc_fake = [[] for i in range(params['K'])]
loss = [[] for i in range(params['K'])]
vars_disc = [[] for i in range(params['K'])]
optim_max = [[] for i in range(params['K'])]
optim_min = [[] for i in range(params['K'])]
opt_max2 = [[] for i in range(params['K'])]
ops_max2 = [[] for i in range(params['K'])]
gradnormsq = [[] for i in range(params['K'])]
grad = [[] for i in range(params['K'])]

optimizer_min = tf.train.AdamOptimizer(params['gen_learning_rate'], beta1=params['beta1'], epsilon=params['epsilon'])
for i in range(params['K']):
    disc_real[i] = discriminator(data,'disc'+str(i),reuse=False)
    disc_fake[i] = discriminator(samples,'disc'+str(i),reuse=True)
    loss[i] = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real[i], labels=tf.ones_like(real_score))) \
        -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake[i], labels=tf.zeros_like(fake_score)))
    vars_disc[i] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'disc'+str(i))
    #optim_min[i] = tf.train.AdamOptimizer(params['gen_learning_rate'], beta1=params['beta1'], epsilon=params['epsilon']).minimize(loss[i],var_list=gen_vars)
    optim_min[i] = optimizer_min.minimize(loss[i],var_list=gen_vars)
    optim_max[i] = tf.train.AdamOptimizer(params['disc_learning_rate'], beta1=params['beta1'], epsilon=params['epsilon']).minimize(-loss[i],var_list=vars_disc[i])
    grad[i] = tf.gradients(loss[i],gen_vars)
    gradnormsq[i] = tf.add_n([tf.reduce_sum(tf.square(g)) for g in grad[i]])


## For eps-steepest descent
al = tf.placeholder(tf.float32, [params['K']])
loss_weighted = tf.reduce_sum(tf.multiply(al, tf.stack(loss,0)))
optim_min_weighted = tf.train.AdamOptimizer(params['gen_learning_rate'], beta1=params['beta1'], epsilon=params['epsilon']).minimize(loss_weighted,var_list=gen_vars)


##################################################################################################
## Begin training

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

ntrial = params['ntrial']
batch_size = params['batch_size']
max_iter = params['max_iter']
viz_every = params['viz_every']
nskip = params['nskip']
nsnap = int(np.ceil(max_iter/np.float(viz_every)))

points = np.nan*np.ones((nsnap,2,batch_size,2,ntrial))
jsd_test = np.nan*np.ones((int(np.ceil(max_iter/np.float(nskip))),ntrial))
time_total = np.nan*np.ones(ntrial)
p_orig = compute_p_orig()

for trial in range(ntrial):
    #print '%d/%d'%(trial,ntrial)
    cnt1 = 0
    cnt2 = 0
    sess.run(tf.global_variables_initializer())
    plt.close(1)
    plt.figure(1)
    time0 = time.time()
    for i in range(max_iter):
        minimax2(sess)
        #gnsq_ = minimax3(sess,eps)
        if i%nskip ==0:
            fs = sess.run(loss)
            id_max = np.argmax(fs) 
            f = np.max(fs)
            jsd = compute_jsd(p_orig)
            jsd_test[cnt1,trial] = jsd
            #jsd_test.append(jsd)
            cnt1 += 1
            print 'trial %d/%d, step %d: jsd=%f, f=%g, id_max=%d'%(trial,ntrial,i,jsd,f,id_max)
            #ids_eps = np.where(fs>=f-eps)[0]
            #print 'step %d: f=%g, id_max=%d, n-eps=%d'%(i,f,id_max,ids_eps.size)
        if i%viz_every == 0:
            xx, yy = sess.run([samples, data])
            points[cnt2,0,:,:,trial] = xx
            points[cnt2,1,:,:,trial] = yy
            cnt2 += 1
            ## Visualize results up to current trial
            ax = plt.subplot(3,np.ceil(nsnap/3.),cnt2)
            ax.set_title('t=%d'%(i),size=12)
            txx = points[cnt2-1,0,:,:,:(trial+1)].transpose((1,0,2)).reshape((2,batch_size*(trial+1)))
            tyy = points[cnt2-1,1,:,:,:(trial+1)].transpose((1,0,2)).reshape((2,batch_size*(trial+1)))    
            plt.scatter(txx[0,:], txx[1,:], c='b', edgecolor='none', s=10, alpha=0.1)
            #plt.axis([-1.5,1.5,-1.5,1.5])
            plt.xlim(-1.5,1.5)
            plt.ylim(-1.5,1.5)
            plt.xticks([])
            plt.yticks([])        
            plt.axis('equal')
            plt.show(block=False)
            plt.pause(3.)  

    time_total[trial] = time.time()-time0
    plt.pause(10.)  


raw_input('Press enter to end')



