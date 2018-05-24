## Same as test_minimax_mog. Just run for K=1 and max_iter=500000

## Parts of codes are from https://github.com/poolio/unrolled_gan

import os
import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from collections import OrderedDict
import tensorflow as tf
ds = tf.contrib.distributions
slim = tf.contrib.slim
import time
from scipy.stats import entropy

import sys
#sys.path.append('/home/hammj/Dropbox/Research/AdversarialNetwork/codes/scripts')
#dir_model = '/home/hammj/Dropbox/Research/AdversarialNetwork/codes/results'
result_dir = '/home/hammj/Dropbox/Research/AdversarialLearning/codes/results/icml18'

#####################################################################################################
## Params
    
params = dict(
    batch_size=128,
    disc_learning_rate=1e-4, # 1e-4
    gen_learning_rate=1e-3, # 1e-3
    beta1=0.5,
    epsilon=1e-8,
    z_dim=256,
    x_dim=2,
    gamma=0,#1E-6,
    K=2, # 1,2,5,10
    max_step=1, # 1,2,5,10
    min_step=1,
    ntrial=10,
    max_iter=100001,
    viz_every=5000,
    nskip=1000,
)

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


#####################################################################################################
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

'''
loglik = mog.log_prob(samples)

def compute_loglik():
    ## Compute log-likelihood
    nrep = 100
    ll = 0.
    for rep in range(nrep):
        #print sess.run(loglik).mean() 
        ll += sess.run(loglik).mean()
    return ll/nrep
'''

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

    
    

##################################################################################################
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
optim_norm_min = [[] for i in range(params['K'])]

optimizer_min = tf.train.AdamOptimizer(params['gen_learning_rate'], beta1=params['beta1'], epsilon=params['epsilon'])#.minimize(loss[i],var_list=gen_vars)
for i in range(params['K']):
    disc_real[i] = discriminator(data,'disc'+str(i),reuse=False)
    disc_fake[i] = discriminator(samples,'disc'+str(i),reuse=True)
    loss[i] = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real[i], labels=tf.ones_like(real_score))) \
        -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake[i], labels=tf.zeros_like(fake_score)))
    vars_disc[i] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'disc'+str(i))
    optim_min[i] = tf.train.AdamOptimizer(params['gen_learning_rate'], beta1=params['beta1'], epsilon=params['epsilon']).minimize(loss[i],var_list=gen_vars)
    optim_max[i] = tf.train.AdamOptimizer(params['disc_learning_rate'], beta1=params['beta1'], epsilon=params['epsilon']).minimize(-loss[i],var_list=vars_disc[i])
    grad[i] = tf.gradients(loss[i],gen_vars)
    gradnormsq[i] = tf.add_n([tf.reduce_sum(tf.square(g)) for g in grad[i]])
    optim_norm_min[i] = optimizer_min.minimize(loss[i] + 0.5*params['gamma']*tf.add_n([tf.reduce_sum(tf.square(g)) for g in tf.gradients(loss[i],vars_disc[i])]),var_list=gen_vars)
    #optim_norm_min[i] = tf.train.AdamOptimizer(params['gen_learning_rate'], beta1=params['beta1'], epsilon=params['epsilon']).minimize(loss[i] + 0.5*params['gamma']*tf.add_n([tf.reduce_sum(tf.square(g)) for g in tf.gradients(loss[i],vars_disc[i])]),var_list=gen_vars)

## For approximate max
lse = tf.placeholder(tf.float32,[],'lse')
loss_approx = lse*tf.reduce_logsumexp(tf.stack(loss,0)/lse)
optim_min_approx = tf.train.AdamOptimizer(params['gen_learning_rate'], beta1=params['beta1'], epsilon=params['epsilon']).minimize(loss_approx,var_list=gen_vars)
#optim_max_approx = tf.train.AdamOptimizer(params['gen_learning_rate'], beta1=params['beta1'], epsilon=params['epsilon']).minimize(-loss_approx,var_list=vars_disc)
optim_max_approx = [[] for i in range(params['K'])]
for i in range(params['K']):
    optim_max_approx[i] = tf.train.AdamOptimizer(params['gen_learning_rate'], beta1=params['beta1'], epsilon=params['epsilon']).minimize(-loss_approx,var_list=vars_disc[i])

## For eps-steepest descent
al = tf.placeholder(tf.float32, [params['K']])
loss_weighted = tf.reduce_sum(tf.multiply(al, tf.stack(loss,0)))
optim_min_weighted = tf.train.AdamOptimizer(params['gen_learning_rate'], beta1=params['beta1'], epsilon=params['epsilon']).minimize(loss_weighted,var_list=gen_vars)


'''
## Copying ops
nvar = len(vars_disc[0])
print nvar
ops_copy=[[[[] for k in range(nvar)] for j in range(params['K'])] for i in range(params['K'])]
for i in range(params['K']):
    for j in range(params['K']):
        for k in range(nvar): # copy from j to i
            ops_copy[i][j][k] = vars_disc[i][k].assign(vars_disc[j][k])
'''


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


def minimax2_norm(sess,feed_dict={}):
    # min step
    for it_min in range(params['min_step']):
        fs = sess.run(loss,feed_dict)
        id_max = np.argmax(fs)
        sess.run(optim_norm_min[id_max],feed_dict)
    # max step
    for it_max in range(params['max_step']):
        sess.run(optim_max,feed_dict)



'''
def minimax2(sess,feed_dict={}):
    al = 0.9
    # min step
    for it_min in range(params['min_step']):
        fs = sess.run(loss,feed_dict)
        minimax2.fs_ = al*np.array(minimax2.fs_) + (1.-al)*np.array(fs)
        id_max = np.argmax(minimax2.fs_) 
        sess.run(optim_min[id_max],feed_dict)
    # max step
    for it_max in range(params['max_step']):
        sess.run(optim_max,feed_dict)
minimax2.fs_ = np.zeros((params['K']))
'''

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


## log-sum-exp
def minimax4(sess, c=1., feed_dict={}):
    #feed_dict['lse'] = c
    # min step
    for it_min in range(params['min_step']):
        sess.run(optim_min_approx,feed_dict={lse:c})
    # max step
    for it_max in range(params['max_step']):
        sess.run(optim_max_approx,feed_dict={lse:c})


##################################################################################################
## Begin training

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

'''
## Efficient graph evaluation?
import time
time0 = time.time()
for i in range(100):
    for k in range(params['K']):
        sess.run(ops_max[k])
print time.time()-time0

time0 = time.time()
for i in range(100):
    sess.run(ops_max)
print time.time()-time0
60s vs 11s
'''



import matplotlib.pyplot as plt

ntrial = params['ntrial']
max_iter = params['max_iter']
viz_every = params['viz_every']
nskip = params['nskip']
#nsnap = (max_iter-1)/viz_every+1
nsnap = int(np.ceil(max_iter/np.float(viz_every)))
points = np.nan*np.ones((nsnap,2,params['batch_size'],2,ntrial))

jsd_test = np.nan*np.ones((int(np.ceil(max_iter/np.float(nskip))),ntrial))
time_total = np.nan*np.ones(ntrial)

print 'p_orig'
p_orig = compute_p_orig()
print 'done'
for trial in range(ntrial):
    print '%d/%d'%(trial,ntrial)
    cnt1 = 0
    cnt2 = 0
    sess.run(tf.global_variables_initializer())
    time0 = time.time()
    for i in range(max_iter):
        minimax2(sess)
        #minimax2_norm(sess)
        #gnsq_ = minimax3(sess,eps)
        #fs.append(f)
        if i%nskip ==0:
            fs = sess.run(loss)
            id_max = np.argmax(fs) 
            f = np.max(fs)
            #p_orig = compute_p_orig()
            jsd = compute_jsd(p_orig)
            jsd_test[cnt1,trial] = jsd
            #jsd_test.append(jsd)
            cnt1 += 1
            #print 'step %d: jsd=%f, f=%g, id_max=%d'%(i,jsd,f,id_max)
            #ids_eps = np.where(fs>=f-eps)[0]
            #print 'step %d: f=%g, id_max=%d, n-eps=%d'%(i,f,id_max,ids_eps.size)
        if i%viz_every == 0:
            xx, yy = sess.run([samples, data])
            points[cnt2,0,:,:,trial] = xx
            points[cnt2,1,:,:,trial] = yy
            cnt2 += 1
    time_total[trial] = time.time()-time0
#print 'time elapsed=%f for ntrial=%d,max_iter=%d,K=%d,J=%d,ga=%8.8f'%(time_total,ntrial,max_iter,params['K'],params['max_step'],params['gamma'])
#print 'time per 1000 iter=%f'%(1000.*time_total/np.float(ntrial*max_iter))

np.save(result_dir+'/points_mog_K%d_J%d_ga%8.8f_al%6.6f.npy'%(params['K'],params['max_step'],params['gamma'],params['gen_learning_rate']),points)
np.save(result_dir+'/jsd_test_mog_K%d_J%d_ga%8.8f_al%6.6f.npy'%(params['K'],params['max_step'],params['gamma'],params['gen_learning_rate']),jsd_test)
np.save(result_dir+'/time_total_mog_K%d_J%d_ga%8.8f_al%6.6f.npy'%(params['K'],params['max_step'],params['gamma'],params['gen_learning_rate']),time_total)

#np.save(result_dir+'/points_mog_K%d_J%d_ga%8.8f_al%6.6f.npy'%(params['K'],params['max_step'],params['gamma'],params['gen_learning_rate']),points)
#np.save(result_dir+'/jsd_test_mog_K%d_J%d_ga%8.8f_al%6.6f.npy'%(params['K'],params['max_step'],params['gamma'],params['gen_learning_rate']),jsd_test)
#np.save(result_dir+'/time_total_mog_K%d_J%d_ga%8.8f_al%6.6f.npy'%(params['K'],params['max_step'],params['gamma'],params['gen_learning_rate']),time_total)

#print jsd_test

#raw_input('Press enter to end')
#######################################################################################################################
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

result_dir = '/home/hammj/Dropbox/Research/AdversarialLearning/codes/results/icml18'

max_iter=100001
viz_every=5000
nskip=1000
ntrial=10
#ts = np.arange(int(np.ceil(max_iter/np.float(nskip))))
ts = np.arange(0,max_iter,nskip)


## No sensitivity penalty

Ks = [1,2,5,10]
Js = [1]#,2,5]#,10]
#gs = [1E-1,1E-2]
jsd_test_all = np.zeros((len(ts),ntrial,len(Ks),len(Js)))
for i,K in enumerate(Ks):
    for j,J in enumerate(Js):
        jsd_test_all[:,:,i,j] = np.load(result_dir+'/jsd_test_mog_K%d_J%d_ga%8.8f_al0.001000.npy'%(K,J,0))        

plt.figure(1)

for i,K in enumerate(Ks):
    for j,J in enumerate(Js):
        #ax = plt.subplot(len(Js),len(Ks),j*len(Ks)+i+1)
        ax = plt.subplot(3,len(Ks),j*len(Ks)+i+1)
        ax.set_title(r'K=%d'%(K),size=18)
        plt.plot(ts,jsd_test_all[:,:,i,j],'y',alpha=0.5)
        tmean = jsd_test_all[:,:,i,j].mean(1).squeeze()
        tstd = jsd_test_all[:,:,i,j].std(1).squeeze()
        plt.fill_between(ts,tmean-tstd,tmean+tstd,where=None,interpolate=True,color='c',alpha=0.5)#,facecolor = )
        plt.plot(ts,tmean,'b',linewidth=2.)
        plt.axis((0,max_iter,0,0.7))
        if i==0:
            plt.ylabel('JS Divergence',size=18)


plt.show(block=False)

with matplotlib.backends.backend_pdf.PdfPages(result_dir+'/jsd_gan_mog.pdf') as pdf:
    pdf.savefig(bbox_inches='tight')



## Plot mog density

## Group by iteration
plt.figure(2)
plt.close()
plt.figure(2)

nsnap = int(np.ceil(max_iter/np.float(viz_every)))
steps = np.arange(0,max_iter,viz_every)

cnt = 0
step = 20
for i,K in enumerate(Ks):
    for j,J in enumerate(Js):
        #ax = plt.subplot(len(Js),len(Ks),j*len(Ks)+i+1)
        ax = plt.subplot(3,len(Ks),j*len(Ks)+i+1)
        #ax.set_title(r'K=%d, J=%d'%(K,J),size=12)
        ax.set_title(r'K=%d'%(K),size=12)
        points = np.load(result_dir+'/points_mog_K%d_J%d_ga%8.8f_al0.001000.npy'%(K,J,0))
        nsnap,_,batch_size,_,ntrial = points.shape
        #plt.tight_layout()
        #for cnt in [nsnap-1]:#range(nsnap):
        xx = points[step,0,:,:,:].transpose((1,0,2)).reshape((2,batch_size*ntrial))
        yy = points[step,1,:,:,:].transpose((1,0,2)).reshape((2,batch_size*ntrial))    
        
        plt.scatter(xx[0,:], xx[1,:], c='b', edgecolor='none', s=10, alpha=0.1)
        #plt.scatter(yy[0,:], yy[1,:], c='g', edgecolor='none', s=10, alpha=0.5)
        plt.axis([-1.5,1.5,-1.5,1.5])
        #plt.axis('equal')
        #plt.title('t=%d'%(steps[cnt]),fontsize=14)
        #if cnt%4==0:
        #    plt.text(-3,0,'J=%d'%(max_step),fontsize=14)                        
        #if cnt<4:
        #    plt.text(-0.5,2,'K=%d'%(K),fontsize=14)
        #plt.axis('off')
        plt.xticks([])
        plt.yticks([])        
        #cnt += 1

plt.show(block=False)

with matplotlib.backends.backend_pdf.PdfPages(result_dir+'/gan_mog_%d.pdf'%(steps[step])) as pdf:
    pdf.savefig(bbox_inches='tight')


## Group by K

K = 10
J = 1

plt.figure(3)
plt.close()
plt.figure(3)

#for i,K in enumerate(Ks):
#for j,J in enumerate(Js):
points = np.load(result_dir+'/points_mog_K%d_J%d_ga%8.8f_al0.001000.npy'%(K,J,0))
nsnap,_,batch_size,_,ntrial = points.shape
#plt.tight_layout()
for k,step in enumerate([2,4,10]):
    ax = plt.subplot(3,4,k+1)
    ax.set_title(r'iter=%d'%(steps[step]),size=12)
    xx = points[step,0,:,:,:].transpose((1,0,2)).reshape((2,batch_size*ntrial))
    #yy = points[step,1,:,:,:].transpose((1,0,2)).reshape((2,batch_size*ntrial))    
    plt.scatter(xx[0,:], xx[1,:], c='b', edgecolor='none', s=10, alpha=0.1)
    plt.axis([-1.5,1.5,-1.5,1.5])
    plt.xticks([])
    plt.yticks([])        

plt.show(block=False)

with matplotlib.backends.backend_pdf.PdfPages(result_dir+'/gan_mog_K%d_J%d.pdf'%(K,J)) as pdf:
    pdf.savefig(bbox_inches='tight')









## With sensitivity penalty

Ks = [1,5,10]
#gs = [1E0,1E-1,1E-2,1E-4,1E-6,0]
gs = [1E-1,1E-2]
jsd_test_all = np.zeros((len(ts),ntrial,len(Ks),len(gs)))
for i,K in enumerate(Ks):
    for j,gamma in enumerate(gs):
        jsd_test_all[:,:,i,j] = np.load(result_dir+'/jsd_test_mog_K%d_J%d_ga%8.8f_al0.000100.npy'%(K,1,gamma))        
        #jsd_test_all[:,:,i,j] = np.load(result_dir+'/jsd_test_mog_K%d_J%d_ga%8.8f.npy'%(K,1,gamma))        
        

#plt.figure(1)
#plt.close()
plt.figure(1)

for i,K in enumerate(Ks):
    for j,gamma in enumerate(gs):
        ax = plt.subplot(len(Ks),len(gs),i*len(gs)+j+1)
        ax.set_title(r'K=%d,$\gamma$=%8.8f'%(K,gamma),size=12)
        plt.plot(ts,jsd_test_all[:,:,i,j],'c',alpha=0.5)
        tmean = jsd_test_all[:,:,i,j].mean(1).squeeze()
        tstd = jsd_test_all[:,:,i,j].std(1).squeeze()
        #plt.plot(ts,tmean+tstd,'g',ts,tmean-tstd,'g')
        plt.fill_between(ts,tmean-tstd,tmean+tstd,where=None,interpolate=True,color='g',alpha=0.5)#,facecolor = )
        plt.plot(ts,tmean,'b')
        #plt.errorbar(ts,jsd_test_all[:,:,i,j].mean(1).squeeze(),yerr=jsd_test_all[:,:,i,j].std(1).squeeze())

plt.show(block=False)

with matplotlib.backends.backend_pdf.PdfPages('temp2.pdf') as pdf:
    pdf.savefig(bbox_inches='tight')



'''

