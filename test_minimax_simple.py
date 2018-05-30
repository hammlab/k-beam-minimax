## Compare different minimax optimization methods for representative 2D surfaces.

import os
import numpy as np
from cvxopt import matrix, solvers
from numpy.linalg import inv
from numpy.linalg import solve
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib import colors as mcolors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from minimax_examples import *


###################################################################################################

def GD(maxiter,u0,v0,eta,fn,args,proj):
    ## gd
    du = u0.size
    dv = v0.size
    us = np.nan*np.ones((maxiter,du))
    vs = np.nan*np.ones((maxiter,dv))
    fs = np.nan*np.ones((maxiter))
    gs = np.nan*np.ones((maxiter))
    us[0,:] = u0
    vs[0,:] = v0
    f,fu,fv,fuu,fuv,fvu,fvv = fn(u0,v0,args)
    fs[0] = f
    gs[0] = l2norm(fu,fv)

    for it in range(maxiter-1):
        if False:#eta==None: # line search
            us[it+1,:],vs[it+1,:] = linesearch_proj(fn,us[it,:],vs[it,:],-fu,fv,args,proj)
        else:
            us[it+1,:],vs[it+1,:] = fixedstepsize_proj(fn,us[it,:],vs[it,:],-fu,fv,eta,args,proj,it+1)
        f,fu,fv,fuu,fuv,fvu,fvv = fn(us[it+1,:],vs[it+1,:],args)
        fs[it+1] = f
        gs[it+1] = l2norm(fu,fv)
    
    return [us,vs,fs,gs]
    

def AltGD(maxiter,u0,v0,eta,fn,args,proj):
    ## alt
    max_step = 1
    min_step = 1

    du = u0.size
    dv = v0.size
    us = np.nan*np.ones((maxiter,du))
    vs = np.nan*np.ones((maxiter,dv))
    fs = np.nan*np.ones((maxiter))
    gs = np.nan*np.ones((maxiter))
    us[0,:] = u0
    vs[0,:] = v0
    f,fu,fv,fuu,fuv,fvu,fvv = fn(u0,v0,args)
    fs[0] = f
    gs[0] = l2norm(fu,fv)
        
    #it = 0
    for it in range(maxiter-1):
        u = us[it,:]
        v = vs[it,:]
        for it_max in range(max_step):
            if False:#eta==None: # line search
                _,v = linesearch_proj(fn,u,v,np.zeros((du)),fv,args,proj)
            else:
                _,v = fixedstepsize_proj(fn,u,v,np.zeros((du)),fv,eta,args,proj,it+1)
            f,fu,fv,fuu,fuv,fvu,fvv = fn(u,v,args)
                
        for it_min in range(min_step):
            if False:#eta==None: # line search
                u,_ = linesearch_proj(fn,u,v,-fu,np.zeros((dv)),args,proj)
            else:
                u,_ = fixedstepsize_proj(fn,u,v,-fu,np.zeros((dv)),eta,args,proj,it+1)
            f,fu,fv,fuu,fuv,fvu,fvv = fn(u,v,args)
            
        us[it+1] = u
        vs[it+1] = v
        fs[it+1] = f
        gs[it+1] = l2norm(fu,fv)
                
    return [us,vs,fs,gs]        
    


## Increasing V, incremental updates
def minimax1(maxiter,nextra,u0,v0,eta,fn,args,proj):

    max_step = 1
    min_step = 1

    du = u0.size
    dv = v0.size
    us = np.nan*np.ones((maxiter,du))
    vs = np.nan*np.ones((maxiter,dv))
    fs = np.nan*np.ones((maxiter))
    gs = np.nan*np.ones((maxiter))
    us[0,:] = u0
    vs[0,:] = v0
    f,fu,fv,fuu,fuv,fvu,fvv = fn(u0,v0,args)
    fs[0] = f
    gs[0] = l2norm(fu,fv)
    
    #Vrand = np.linspace(-0.5,0.5,nextra)
    #np.random.shuffle(Vrand)    
    Vrand = np.random.rand
    for it in range(maxiter-1):
        ## min step:
        # True: min_u max_{1<=i<=k} f(u,v_i), Approx: min_u log-sum-exp(f(u,v_i))
        # nabla_u log-sum-exp(f(u,v_i)) = sum_i [exp(fi) nabla_u fi]/(sum_i exp(fi))
        # A few steps of gradient descent updates
        # Directional derivative 
        # Dw (max_i fi(u)) = max_i <nabla fi(u),w>
        # nabla (max_i fi(u))_j = Dej (...) = max_i <nabla fi(u),e_j>
        u = us[it,:]
        #v = vs[it,:]
        for it_min in range(min_step):
            fus = np.zeros((it+1+nextra,du))
            fvals = np.zeros((it+1+nextra))
            for i in range(it+1): # max among all vi and Vrand
                f,fu,_,_,_,_,_ = fn(u,vs[i,:],args)
                fvals[i] = f
                fus[i,:] = fu
            for i in range(nextra):
                f,fu,_,_,_,_,_ = fn(u,Vrand[i],args)
                fvals[it+1+i] = f
                fus[it+1+i,:] = fu
            indmax = fvals.argmax(axis=0)
            gmax = fus[indmax,:]
            if indmax>it:
                vmax = Vrand[indmax-it-1]
            else :
                vmax = vs[indmax]
            if False:#eta==None: # line search
                u,_ = linesearch_proj(fn,u,vmax,-gmax,np.zeros((dv)),args,proj)
            else:
                u,_ = fixedstepsize_proj(fn,u,vmax,-gmax,np.zeros((dv)),eta,args,proj,it+1)
    
        ## max step:
        # vmax = argmax_v\inVi f(ui,v)
        # v(i+1) = vmax + \eta*df(ui,vmax)/dv
        for it_max in range(max_step):
            fvs = np.zeros((it+1+nextra,du))
            fvals = np.zeros((it+1+nextra))
            for i in range(it+1): # max among all vi and Vrand
                f,_,fv,_,_,_,_ = fn(u,vs[i,:],args)
                fvals[i] = f
                fvs[i,:] = fv
            for i in range(nextra):
                f,_,fv,_,_,_,_ = fn(u,Vrand[i],args)
                fvals[it+1+i] = f
                fvs[it+1+i,:] = fv
            indmax = fvals.argmax(axis=0)
            gmax = fvs[indmax,:]
            if indmax>it:
                vmax = Vrand[indmax-it-1]
            else :
                vmax = vs[indmax]
            if False:#eta==None: # line search
                _,v = linesearch_proj(fn,u,vmax,np.zeros((du)),gmax,args,proj)
            else:
                _,v = fixedstepsize_proj(fn,u,vmax,np.zeros((du)),gmax,eta,args,proj,it+1)

        f,fu,fv,fuu,fuv,fvu,fvv = fn(u,v,args)
        us[it+1] = u
        vs[it+1] = v
        fs[it+1] = f
        gs[it+1] = l2norm(fu,fv)
    
    return [us,vs,fs,gs]        


## Fixed V, incremental updates
def minimax2(maxiter,K,u0,V,eta,fn,args,proj):

    max_step = 1
    min_step = 1
    v0 = V[0]
    du = u0.size
    dv = v0.size
    us = np.nan*np.ones((maxiter,du))
    vs = np.nan*np.ones((maxiter,dv))
    fs = np.nan*np.ones((maxiter))
    gs = np.nan*np.ones((maxiter))
    us[0,:] = u0
    vs[0,:] = v0
    f,fu,fv,fuu,fuv,fvu,fvv = fn(u0,v0,args)
    fs[0] = f
    gs[0] = l2norm(fu,fv)
    
    #np.random.shuffle(V)
    for it in range(maxiter-1):
        # Directional derivative 
        # Dw (max_i fi(u)) = max_i <nabla fi(u),w>
        # nabla (max_i fi(u))_j = Dej (...) = max_i <nabla fi(u),e_j>
        u = us[it,:]
        for it_min in range(min_step):
            fus = np.zeros((K,du))
            fvals = np.zeros((K))
            for k in range(K): # max among V
                f,fu,_,_,_,_,_ = fn(u,V[k],args)
                fvals[k] = f
                fus[k,:] = fu
            indmax = fvals.argmax(axis=0)
            gmax = fus[indmax,:]
            vmax = V[indmax]
            if False:#eta==None: # line search
                u,_ = linesearch_proj(fn,u,vmax,-gmax,np.zeros((dv)),args,proj)
            else:
                u,_ = fixedstepsize_proj(fn,u,vmax,-gmax,np.zeros((dv)),eta,args,proj,it+1)
        
        ## max step:
        # For k in 1:K, v(k,i+1) = v(k,i) + \eta(k,i) df(uk,v(k,i))/dv
        for it_max in range(max_step):
            fvs = np.zeros((K,du))
            fvals = np.zeros((K))
            for k in range(K): # max among V
                f,_,fv,_,_,_,_ = fn(u,V[k],args)
                fvals[k] = f
                fvs[k,:] = fv
            indmax = fvals.argmax(axis=0)
            # Update all V's
            for k in range(K):
                if False:#eta==None: # line search
                    _,V[k] = linesearch_proj(fn,u,V[k],np.zeros((du)),fvs[k,:],args,proj)
                else:
                    _,V[k] = fixedstepsize_proj(fn,u,V[k],np.zeros((du)),fvs[k,:],eta,args,proj,it+1)
        
        f,fu,fv,fuu,fuv,fvu,fvv = fn(u,V[indmax],args)
        us[it+1] = u
        vs[it+1] = V[indmax] # This doesn't really matter.
        fs[it+1] = f
        gs[it+1] = l2norm(fu,fv)
    
    return [us,vs,fs,gs]        



## epsilon-steepest descent
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
    if np.random.randn(1)>0.95:
        if any(np.dot(Z,z.T)<0):
            print np.dot(Z,z.T)
            print 'negative angle!!!!'
            asdfasd
    return [sol['x'],z]#sol['primal objective']]


def minimax3(maxiter,K,u0,V,eta,fn,args,proj):

    max_step = 1
    min_step = 1
    v0 = V[0]
    du = u0.size
    dv = v0.size
    us = np.nan*np.ones((maxiter,du))
    vs = np.nan*np.ones((maxiter,dv))
    fs = np.nan*np.ones((maxiter))
    gs = np.nan*np.ones((maxiter))
    us[0,:] = u0
    vs[0,:] = v0
    f,fu,fv,fuu,fuv,fvu,fvv = fn(u0,v0,args)
    fs[0] = f
    gs[0] = l2norm(fu,fv)
    
    #np.random.shuffle(V)
    for it in range(maxiter-1):
        # Directional derivative 
        # Dw (max_i fi(u)) = max_i <nabla fi(u),w>
        # nabla (max_i fi(u))_j = Dej (...) = max_i <nabla fi(u),e_j>
        u = us[it,:]
        for it_min in range(min_step):
            fus = np.zeros((K,du))
            fvals = np.zeros((K))
            for k in range(K): # max among V
                f,fu,_,_,_,_,_ = fn(u,V[k],args)
                fvals[k] = f
                fus[k,:] = fu
            indmax = fvals.argmax(axis=0)
            #gmax = fus[indmax,:]
            #vmax = V[indmax]
       
            ids_eps = np.where(fvals>=fvals[indmax]-eps)[0]
            if len(ids_eps)==1:
                sess.run(optim_min[ids_eps[0]],feed_dict)
                gnsq = sess.run(gradnormsq[ids_eps[0]],feed_dict)            
            else:
                # Solve QP
                a, z = EpsSteepestDescentDirection(fus[ids_eps])
                #gmax = z/np.sqrt(np.sum(z**2))
                gmax = z
                #print np.sqrt(gnsq)
                if False:#eta==None: # line search
                    u,_ = linesearch_proj(fn,u,v0,-gmax,np.zeros((dv)),args,proj)
                else:
                    u,_ = fixedstepsize_proj(fn,u,v0,-gmax,np.zeros((dv)),eta,args,proj,it+1)
        
        ## max step:
        # For k in 1:K, v(k,i+1) = v(k,i) + \eta(k,i) df(uk,v(k,i))/dv
        for it_max in range(max_step):
            fvs = np.zeros((K,du))
            fvals = np.zeros((K))
            for k in range(K): # max among V
                f,_,fv,_,_,_,_ = fn(u,V[k],args)
                fvals[k] = f
                fvs[k,:] = fv
            indmax = fvals.argmax(axis=0)
            # Update all V's
            for k in range(K):
                if False:#eta==None: # line search
                    _,V[k] = linesearch_proj(fn,u,V[k],np.zeros((du)),fvs[k,:],args,proj)
                else:
                    _,V[k] = fixedstepsize_proj(fn,u,V[k],np.zeros((du)),fvs[k,:],eta,args,proj,it+1)
        
        f,fu,fv,fuu,fuv,fvu,fvv = fn(u,V[indmax],args)
        us[it+1] = u
        vs[it+1] = V[indmax] # This doesn't really matter.
        fs[it+1] = f
        gs[it+1] = l2norm(fu,fv)
    
    return [us,vs,fs,gs]        


def proj_linf(u,v,th=0.5):
    u = np.clip(u,-th,th)#np.maximum(u,-2)
    v = np.clip(v,-th,th)
    return [u,v]


def l2norm(u,v):
    return np.sqrt((u**2).sum()+(v**2).sum())


def fixedstepsize_proj(fn,u0,v0,delu,delv,eta,args,proj,t=1.):
    #r = np.sqrt(delu**2 + delv**2)
    #delu /= r
    #delv /= r
    if True:
        u_ = u0 + eta/np.sqrt(t)*delu
        v_ = v0 + eta/np.sqrt(t)*delv
    else:
        u_ = u0 + eta/t*delu
        v_ = v0 + eta/t*delv
    u_,v_ = proj(u_,v_)
    return u_,v_



####################################################################################################
## Compare sgd, alt, gradnorm for fixed learning rates
# Examples of saddle points = f1, f2, f8
# Examples of no saddle points = f32, f12, f9


eta = 1E-1
ntrial = 100
maxiter = 1000
labels=['GD','Alt-GD','minimax K=1','minimax K=2','minimax K=5','minimax K=10']
nmethods = len(labels)

errs_norm = np.nan*np.ones((6,ntrial,maxiter,nmethods))

for i,fn in enumerate([f1,f2,f8,f32,f12,f9]):
    print '\n\nExample %d/%d'%(i,6)
    args = np.zeros(0)
    proj = proj_linf
    if i==3:
        uast = np.array([-0.25,0.25])
    else:
        uast = np.array([0.])

    for trial in range(ntrial):
        #print 'Trial %d/%d'%(trial,ntrial)
        u0 = 0.45*(2.*np.random.rand(1)-1.)
        V = 0.45*(2*np.random.rand(10)-1.)
        v0 = V[0]
        #u0,v0 = proj(u0,v0)
        nextra = 0
        us1,vs1,fs1,gs1 = GD(maxiter,u0,v0,eta,fn,args,proj)    
        us2,vs2,fs2,gs2 = AltGD(maxiter,u0,v0,eta,fn,args,proj)
        us3,vs3,fs3,gs3 = minimax2(maxiter,1,u0,[v0],eta,fn,args,proj)
        us4,vs4,fs4,gs4 = minimax2(maxiter,2,u0,V[:2],eta,fn,args,proj)
        us5,vs5,fs5,gs5 = minimax2(maxiter,5,u0,V[:5],eta,fn,args,proj)
        us6,vs6,fs6,gs6 = minimax2(maxiter,10,u0,V,eta,fn,args,proj)
        
        uss=[us1,us2,us3,us4,us5,us6]
        vss=[vs1,vs2,vs3,vs4,vs5,vs6]
        fss=[fs1,fs2,fs3,fs4,fs5,fs6]
        gss=[gs1,gs2,gs3,gs4,gs5,gs6]

        if i==3:        
            errs_norm[i,trial,:,0] = np.minimum(np.abs(us1-uast[0]),np.abs(us1-uast[1])).reshape((maxiter))
            errs_norm[i,trial,:,1] = np.minimum(np.abs(us2-uast[0]),np.abs(us2-uast[1])).reshape((maxiter))
            errs_norm[i,trial,:,2] = np.minimum(np.abs(us3-uast[0]),np.abs(us3-uast[1])).reshape((maxiter))
            errs_norm[i,trial,:,3] = np.minimum(np.abs(us4-uast[0]),np.abs(us4-uast[1])).reshape((maxiter))
            errs_norm[i,trial,:,4] = np.minimum(np.abs(us5-uast[0]),np.abs(us5-uast[1])).reshape((maxiter))
            errs_norm[i,trial,:,5] = np.minimum(np.abs(us6-uast[0]),np.abs(us6-uast[1])).reshape((maxiter))
        else:
            errs_norm[i,trial,:,0] = np.abs(us1-uast[0]).squeeze()
            errs_norm[i,trial,:,1] = np.abs(us2-uast[0]).squeeze()
            errs_norm[i,trial,:,2] = np.abs(us3-uast[0]).squeeze()
            errs_norm[i,trial,:,3] = np.abs(us4-uast[0]).squeeze()
            errs_norm[i,trial,:,4] = np.abs(us5-uast[0]).squeeze()
            errs_norm[i,trial,:,5] = np.abs(us6-uast[0]).squeeze()
            
        tmean = errs_norm[i,:(trial+1),-1,:].mean(0).squeeze()
        print 'trial %d/%d, test error: %4.3f (%s), %4.3f (%s), %4.3f (%s), %4.3f (%s), %4.3f (%s), %4.3f (%s)'%(trial,ntrial,tmean[0],labels[0],tmean[1],labels[1],tmean[2],labels[2],tmean[3],labels[3],tmean[4],labels[4],tmean[5],labels[5]) 

                    

tmean = errs_norm[:,:,-1,:].mean(1).squeeze()
tstd = errs_norm[:,:,-1,:].std(1).squeeze()

for i in range(6):
    print '\\'
    for j in range(nmethods):
        print '%4.3f \\pm %4.3f &'%(tmean[i,j],tstd[i,j]),





