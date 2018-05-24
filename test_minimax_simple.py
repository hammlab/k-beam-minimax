## Compare different optimization methods for several saddle point problems.

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
#from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#import tensorflow as tf


from minimax_examples import *

result_dir = '/home/hammj/Dropbox/Research/AdversarialLearning/codes/results/icml18'

###################################################################################################

def GD(max_iter,u0,v0,eta,fn,args,proj):
    ## gd
    du = u0.size
    dv = v0.size
    us = np.nan*np.ones((max_iter,du))
    vs = np.nan*np.ones((max_iter,dv))
    fs = np.nan*np.ones((max_iter))
    gs = np.nan*np.ones((max_iter))
    us[0,:] = u0
    vs[0,:] = v0
    f,fu,fv,fuu,fuv,fvu,fvv = fn(u0,v0,args)
    fs[0] = f
    gs[0] = l2norm(fu,fv)

    for it in range(max_iter-1):
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
    

def NewtGD2(max_iter,u0,v0,eta,fn,args,proj):
    ## newt-gd
    # min: one sgd step for f + eta*\|grad_v f(u,v)\|^2
    du = u0.size
    dv = v0.size
    us = np.nan*np.ones((max_iter,du))
    vs = np.nan*np.ones((max_iter,dv))
    fs = np.nan*np.ones((max_iter))
    gs = np.nan*np.ones((max_iter))
    us[0,:] = u0
    vs[0,:] = v0
    f,fu,fv,fuu,fuv,fvu,fvv = fn(u0,v0,args)
    fs[0] = f
    gs[0] = l2norm(fu,fv)

    for it in range(max_iter-1):
        #f,fu,fv,fuu,fuv,fvu,fvv = fn(us[it,:],vs[it,:],args)
        #if eta==None: # line search
        u_ = us[it,:] - eta*(fu + eta*np.dot(fuv,fv))
        v_ = vs[it,:] + eta*(fv - eta*np.dot(fvu,fu))
        us[it+1,:],vs[it+1,:] = proj(u_,v_)    
        f,fu,fv,fuu,fuv,fvu,fvv = fn(us[it+1,:],vs[it+1,:],args)
        fs[it+1] = f
        gs[it+1] = l2norm(fu,fv)

    return [us,vs,fs,gs]        
    
def NewtGD1(max_iter,u0,v0,eta,fn,args,proj):
    du = u0.size
    dv = v0.size
    us = np.nan*np.ones((max_iter,du))
    vs = np.nan*np.ones((max_iter,dv))
    fs = np.nan*np.ones((max_iter))
    gs = np.nan*np.ones((max_iter))
    us[0,:] = u0
    vs[0,:] = v0
    f,fu,fv,fuu,fuv,fvu,fvv = fn(u0,v0,args)
    fs[0] = f
    gs[0] = l2norm(fu,fv)

    etav = eta
    etau = eta
    for it in range(max_iter-1):
        #f,fu,fv,fuu,fuv,fvu,fvv = fn(us[it,:],vs[it,:],args)
        u_ = us[it,:] - eta*fu + 0.5*(eta**2)*(np.dot(fuu,fu)-np.dot(fuv,fv))
        v_ = vs[it,:] + eta*fv + 0.5*(eta**2)*(-np.dot(fvu,fu)+np.dot(fvv,fv))
        us[it+1,:],vs[it+1,:] = proj(u_,v_)    
        f,fu,fv,fuu,fuv,fvu,fvv = fn(us[it+1,:],vs[it+1,:],args)
        fs[it+1] = f
        gs[it+1] = l2norm(fu,fv)

    return [us,vs,fs,gs]        


def Newton(max_iter,u0,v0,fn,args,proj):
    ## Newton1
    du = u0.size
    dv = v0.size
    us = np.nan*np.ones((max_iter,du))
    vs = np.nan*np.ones((max_iter,dv))
    fs = np.nan*np.ones((max_iter))
    gs = np.nan*np.ones((max_iter))
    us[0,:] = u0
    vs[0,:] = v0
    f,fu,fv,fuu,fuv,fvu,fvv = fn(u0,v0,args)
    fs[0] = f
    gs[0] = l2norm(fu,fv)

    H = np.zeros((du+dv,du+dv))
    fufv = np.zeros((du+dv))
    for it in range(max_iter-1):
        #f,fu,fv,fuu,fuv,fvu,fvv = fn(us[it,:],vs[it,:],args)
        #H*x = -fx    
        H[:du,:du] = fuu
        H[:du,du:] = fuv
        H[du:,:du] = fvu
        H[du:,du:] = fvv
        fufv[:du] = fu
        fufv[du:] = fv
        Hinvdf = solve(H,fufv)#  
        #Hinv = inv(H)
        #uv = Hinv.dot(np.concatenate((fu,fv),0))
        
        if False:#eta==None: # line search
            us[it+1,:],vs[it+1,:] = linesearch_proj(fn,us[it,:],vs[it,:],-Hinvdf[:du],-Hinvdf[du:],args,proj)
        else:
            us[it+1,:],vs[it+1,:] = fixedstepsize_proj(fn,us[it,:],vs[it,:],-Hinvdf[:du],-Hinvdf[du:],1.0,args,proj)
        f,fu,fv,fuu,fuv,fvu,fvv = fn(us[it+1,:],vs[it+1,:],args)
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


## Fixed V, incremental updates
def minimax2_norm(maxiter,K,u0,V,eta,fn,args,proj):

    ga = 1E-3
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
                f,fu,fv,fuu,fuv,_,_ = fn(u,V[k],args)
                fvals[k] = f
                fus[k,:] = fu + ga*fuv*fv
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
    if np.random.randn(1)>0.95:
        if any(np.dot(Z,z.T)<0):
            print np.dot(Z,z.T)
            print 'negative angle!!!!'
            asdfasd
    return [sol['x'],z]#sol['primal objective']]



## epsilon-steepest descent
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





## Fixed V, overwrite the worst one
def minimax4(maxiter,K,u0,v0,eta,fn,args,proj):

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
    
    V = np.linspace(-0.5,0.5,K)
    np.random.shuffle(V)
    scores = np.zeros((K),np.float) # of times it dominated 
    for it in range(maxiter-1):
        ## min step:
        # True: min_u max_{1<=i<=k} f(u,v_i), Approx: min_u log-sum-exp(f(u,v_i))
        # nabla_u log-sum-exp(f(u,v_i)) = sum_i [exp(fi) nabla_u fi]/(sum_i exp(fi))
        # A few steps of gradient descent updates
        # Directional derivative 
        # Dw (max_i fi(u)) = max_i <nabla fi(u),w>
        # nabla (max_i fi(u))_j = Dej (...) = max_i <nabla fi(u),e_j>
        u = us[it,:]
        v = vs[it,:]
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
        # vmax = argmax_{v\inVi} f(ui,v)
        # v(k+1) = argmax_v f(uk,v)
        for it_max in range(max_step):
            fvs = np.zeros((K,du))
            fvals = np.zeros((K))
            for k in range(K): # max among V
                f,_,fv,_,_,_,_ = fn(u,V[k],args)
                fvals[k] = f
                fvs[k,:] = fv
            indmax = fvals.argmax(axis=0)
            gmax = fvs[indmax,:]
            vmax = V[indmax]
            # f(uk,v) >= max_V f(uk,v)
            if False:#eta==None: # line search
                _,v = linesearch_proj(fn,u,vmax,np.zeros((du)),gmax,args,proj)
            else:
                _,v = fixedstepsize_proj(fn,u,vmax,np.zeros((du)),gmax,eta,args,proj,it+1)
            # Overwrite the worst one at random?
            scores[indmax] += 1
            indmin = scores.argmin()#fvals.argmin(axis=0)
            V[indmin] = v
            scores[indmin] = scores[indmax]# + 1

        f,fu,fv,fuu,fuv,fvu,fvv = fn(u,v,args)
        us[it+1] = u
        vs[it+1] = v
        fs[it+1] = f
        gs[it+1] = l2norm(fu,fv)
    
    return [us,vs,fs,gs]        




###################################################################################################

def proj_l2(u,v,th=1.0):
    sc = np.sqrt((u**2).sum() + (v**2).sum())
    if sc>th: 
        u=th/sc*u
        v=th/sc*v
    return [u,v]

def proj_linf(u,v,th=0.5):
    u = np.clip(u,-th,th)#np.maximum(u,-2)
    v = np.clip(v,-th,th)
    return [u,v]


def l2norm(u,v):
    return np.sqrt((u**2).sum()+(v**2).sum())

def normalized(x):
    return x
    #return x/np.sqrt((x**2).sum()+1E-20)

'''
def linesearch(fn,u0,v0,delu,delv,args,al=0.01,be=.9):
    t = 1.0
    f0,fu,fv,fuu,fuv,fvu,fvv = fn(u0,v0,args)
    tol = al*(np.dot(fu,delu)+np.dot(fv,delv))
    while (fn(u0+t*delu,v0+t*delv,args)[0] > f0 + t*tol):
        t *= be
    print t
    return t

def linesearch_proj(fn,u0,v0,delu,delv,args,proj,al=0.01,be=.9):
    u_,v_ = proj(u0+delu,v0+delv)
    delu = u_ - u0
    delv = v_ - v0
    t = linesearch(fn,u0,v0,delu,delv,args)
    u_ = u0 + t*delu
    v_ = v0 + t*delv
    return u_,v_
'''

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



############################################################################################################

def PlotSurface(uss,vss,fss,gss,specs,labels,fn,args,proj):

    ## Plot the trajectory
    if du>1 or dv>1:
        print 'du and dv should both be 1'
        return

    # Make surfce
    U,V = np.meshgrid(np.linspace(-0.5,0.5,100),np.linspace(-0.5,0.5,100))
    #U,V = np.meshgrid(np.linspace(-np.sqrt(0.5),np.sqrt(0.5),100),np.linspace(-np.sqrt(0.5),np.sqrt(0.5),100))
    Z = np.zeros(U.shape)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            U[i,j],V[i,j] = proj_linf(U[i,j],V[i,j])
            Z[i,j] = fn(U[i,j],V[i,j],args)[0]

    fig = plt.figure(1)
    #plt.close()
    #fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.plot_surface(U[:,:],V[:,:],Z[:,:],\
    #cmap=cm.coolwarm,rstride=10,cstride=10,linewidth=0)#,edgecolors='tab:gray')#, antialiased=False)
    ax.plot_wireframe(U,V,Z,rstride=10,cstride=10,linewidth=0.5,color='k')#gray')

    nlines = len(uss)
    for i in range(nlines):
        ax.plot(uss[i],vss[i],fss[i],specs[i],label=labels[i],markersize=6,linewidth=3,markeredgewidth=2)

    ax.view_init(elev=60, azim=-45) #Reproduce view
    ax.set_xlabel('U',fontsize=18)
    ax.set_ylabel('V',fontsize=18)
    plt.legend(fontsize=18, loc='upper right')
    plt.show(block=False)

    #with matplotlib.backends.backend_pdf.PdfPages(os.path.join(result_dir,'compare_optim_methods2.pdf')) as pdf:
    #    pdf.savefig(bbox_inches='tight')


def PlotTrajectory(gss,specs,labels):

    ## Plot the trajectory
    fig = plt.figure(2)
    nlines = len(gss)
    max_iter = gss[0].size

    for i in range(nlines):    
        plt.plot(range(max_iter),gss[i],specs[i],label=labels[i],markersize=6,linewidth=2)

    plt.xlabel('iter',fontsize=12)
    #plt.ylabel('gradient norm',fontsize=12)
    #plt.axis([-0.1, 4.1, 0.5, 1.])
    plt.legend(fontsize=18, loc='upper right')
    plt.show(block=False)#False)

    #with matplotlib.backends.backend_pdf.PdfPages(os.path.join(result_dir,'dann_surface_2.pdf')) as pdf:
    #    pdf.savefig(bbox_inches='tight')

def PlotTrajectoryAll(errs,specs,labels):

    fig = plt.figure(2)
    ntrial,max_iter,nlines = errs.shape
    for i in range(ntrial):
        for j in range(nlines):    
            if i==0:
                plt.plot(np.arange(max_iter),errs[i,:,j],specs[j],label=labels[j],markersize=1,linewidth=1)
            else:
                plt.plot(np.arange(max_iter),errs[i,:,j],specs[j],markersize=1,linewidth=1)
                

    plt.xlabel('iter',fontsize=12)
    plt.legend(fontsize=18, loc='upper right')
    plt.show(block=False)#False)



def PlotErr(errs,specs,labels):

    ## Plot the trajectory
    #plt.close()
    #plt.figure()
    fig = plt.figure(3)
    ntrial,max_iter,nlines = errs.shape

    mean_err = errs.mean(0)
    std_err = errs.std(0)
    
    for i in range(nlines):    
        #plt.plot(np.arange(max_iter),mean_err[:,i],specs[i],label=labels[i],markersize=6,linewidth=2)
        plt.errorbar(np.arange(max_iter),mean_err[:,i],yerr=std_err[:,i],marker=specs[i][2],label=labels[i],markersize=6,linewidth=2)        
    
    plt.xlabel('iter',fontsize=12)
    #plt.ylabel('gradient norm',fontsize=12)
    #plt.axis([-0.1, 4.1, 0.5, 1.])
    plt.legend(fontsize=18, loc='upper right')
    plt.show(block=False)#False)


####################################################################################################
## Compare sgd, alt, gradnorm for fixed learning rates

# Examples of saddle points = f1, f2, f8
# Examples of no saddle points = f32, f12, f9

eta = 1E-1

ntrial = 100
max_iter = 1000
labels=['GD','Alt-GD','minimax K=1','minimax K=2','minimax K=5','minimax K=10']
nmethods = len(labels)

errs_norm = np.nan*np.ones((6,ntrial,max_iter,nmethods))
du = 1#00
dv = du#+1

for i,fn in enumerate([f1,f2,f8,f32,f12,f9]):
    print '%d/%d'%(i,6)
    args = np.zeros(0)
    proj = proj_linf
    if i==3:
        uast = np.array([-0.25,0.25])
    else:
        uast = np.array([0.])
    '''
    fn = f4
    A = 0.2*np.random.randn(du,dv)
    b = 0.2*np.random.randn(du) 
    c = 0.2*np.random.randn(dv)
    args = [A,b,c]
    proj = proj_l2
    '''
    for trial in range(ntrial):
        u0 = 0.45*(2.*np.random.rand(1)-1.)
        V = 0.45*(2*np.random.rand(10)-1.)
        v0 = V[0]
        #u0,v0 = proj(u0,v0)
        nextra = 0
        us1,vs1,fs1,gs1 = GD(max_iter,u0,v0,eta,fn,args,proj)    
        us2,vs2,fs2,gs2 = AltGD(max_iter,u0,v0,eta,fn,args,proj)
        us3,vs3,fs3,gs3 = minimax2(max_iter,1,u0,[v0],eta,fn,args,proj)
        us4,vs4,fs4,gs4 = minimax2(max_iter,2,u0,V[:2],eta,fn,args,proj)
        us5,vs5,fs5,gs5 = minimax2(max_iter,5,u0,V[:5],eta,fn,args,proj)
        us6,vs6,fs6,gs6 = minimax2(max_iter,10,u0,V,eta,fn,args,proj)
        
        uss=[us1,us2,us3,us4,us5,us6]#,us7,us8]
        vss=[vs1,vs2,vs3,vs4,vs5,vs6]#,vs7,vs8]
        fss=[fs1,fs2,fs3,fs4,fs5,fs6]#,fs7,fs8]
        gss=[gs1,gs2,gs3,gs4,gs5,gs6]#,gs7,gs8]

        if i==3:        
            errs_norm[i,trial,:,0] = np.minimum(np.abs(us1-uast[0]),np.abs(us1-uast[1])).reshape((max_iter))
            errs_norm[i,trial,:,1] = np.minimum(np.abs(us2-uast[0]),np.abs(us2-uast[1])).reshape((max_iter))
            errs_norm[i,trial,:,2] = np.minimum(np.abs(us3-uast[0]),np.abs(us3-uast[1])).reshape((max_iter))
            errs_norm[i,trial,:,3] = np.minimum(np.abs(us4-uast[0]),np.abs(us4-uast[1])).reshape((max_iter))
            errs_norm[i,trial,:,4] = np.minimum(np.abs(us5-uast[0]),np.abs(us5-uast[1])).reshape((max_iter))
            errs_norm[i,trial,:,5] = np.minimum(np.abs(us6-uast[0]),np.abs(us6-uast[1])).reshape((max_iter))
        else:
            errs_norm[i,trial,:,0] = np.abs(us1-uast[0]).squeeze()
            errs_norm[i,trial,:,1] = np.abs(us2-uast[0]).squeeze()
            errs_norm[i,trial,:,2] = np.abs(us3-uast[0]).squeeze()
            errs_norm[i,trial,:,3] = np.abs(us4-uast[0]).squeeze()
            errs_norm[i,trial,:,4] = np.abs(us5-uast[0]).squeeze()
            errs_norm[i,trial,:,5] = np.abs(us6-uast[0]).squeeze()
                    
        #specs=['c-d','m-^','b-v','g-s','r-o','p-]
        #specs=['c-d','m-^','b-v','b-s','b-o','r-v','r-s','r-o',]
        
        if False:
            PlotSurface(uss,vss,fss,gss,specs,labels,fn,args,proj)
            #PlotTrajectory(uss,specs,labels)
            PlotTrajectory(errs_norm[i,trial,:,:].squeeze().transpose(),specs,labels)            
            raw_input('Press enter to continue')
            plt.figure(1)
            plt.close()
            plt.figure(2)
            plt.close()
    
#PlotTrajectoryAll(errs_norm,specs,labels)
np.save(result_dir+'/errs_norm.npy',errs_norm)


## Convergence rates
ths = [0.1,0.05,0.02,0.01]
convrates = np.nan*np.ones((len(ths),6,nmethods)) # nthresholds x 6 surfaces x nmethods

for i in range(6):
    for j in range(len(ths)):
        for k in range(nmethods):    
            convrates[j,i,k] = len(np.where(errs_norm[i,:,-1,k].squeeze()<ths[j])[0])/np.float(ntrial)
            #convrates[j,i,k] = len(np.where(errs_norm[i,:,:,k].min(1).squeeze()<ths[j])[0])/np.float(ntrial)            

print convrates

np.save(result_dir+'/convrates.npy',convrates)

#raw_input('Press enter to continue')

#with matplotlib.backends.backend_pdf.PdfPages(os.path.join(result_dir,'compare_simple.pdf')) as pdf:
    #pdf.savefig(bbox_inches='tight')

#######################################################################################################

'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib import colors as mcolors
from matplotlib.ticker import LinearLocator, FormatStrFormatter

result_dir = '/home/hammj/Dropbox/Research/AdversarialLearning/codes/results/icml18'

errs_norm = np.load(result_dir+'/errs_norm.npy')
nsurfaces,ntrial,max_iter,nmethods = errs_norm.shape
convrates = np.load(result_dir+'/convrates.npy')

'''


## Error after max_iter

tmean = errs_norm[:,:,-1,:].mean(1).squeeze()
tstd = errs_norm[:,:,-1,:].std(1).squeeze()

for i in range(6):
    print '\\'
    for j in range(nmethods):
        print '%4.3f \\pm %4.3f &'%(tmean[i,j],tstd[i,j]),

'''
# 5E-2
0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & \
0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & \
0.216 \pm 0.123 & 0.207 \pm 0.121 & 0.208 \pm 0.121 & 0.030 \pm 0.080 & 0.003 \pm 0.017 & \
0.217 \pm 0.027 & 0.216 \pm 0.027 & 0.216 \pm 0.027 & 0.115 \pm 0.106 & 0.029 \pm 0.072 & \
0.500 \pm 0.000 & 0.499 \pm 0.008 & 0.499 \pm 0.008 & 0.315 \pm 0.241 & 0.091 \pm 0.192 & \
0.109 \pm 0.054 & 0.109 \pm 0.054 & 0.109 \pm 0.054 & 0.048 \pm 0.058 & 0.001 \pm 0.007 &

# 1E-1
\
0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & \
0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & \
0.261 \pm 0.123 & 0.223 \pm 0.117 & 0.213 \pm 0.111 & 0.026 \pm 0.079 & 0.005 \pm 0.038 & 0.002 \pm 0.001 & \
0.242 \pm 0.007 & 0.242 \pm 0.007 & 0.242 \pm 0.007 & 0.107 \pm 0.117 & 0.044 \pm 0.090 & 0.006 \pm 0.031 & \
0.500 \pm 0.000 & 0.500 \pm 0.000 & 0.500 \pm 0.000 & 0.306 \pm 0.243 & 0.047 \pm 0.143 & 0.007 \pm 0.050 & \
0.139 \pm 0.041 & 0.139 \pm 0.041 & 0.140 \pm 0.039 & 0.018 \pm 0.045 & 0.001 \pm 0.000 & 0.001 \pm 0.000 &

0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & \
0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & \
0.258 \pm 0.123 & 0.215 \pm 0.130 & 0.215 \pm 0.131 & 0.045 \pm 0.102 & 0.003 \pm 0.016 & \
0.244 \pm 0.004 & 0.244 \pm 0.004 & 0.244 \pm 0.004 & 0.128 \pm 0.115 & 0.039 \pm 0.086 & \
0.500 \pm 0.000 & 0.500 \pm 0.000 & 0.500 \pm 0.000 & 0.291 \pm 0.246 & 0.042 \pm 0.135 & \
0.131 \pm 0.045 & 0.131 \pm 0.045 & 0.131 \pm 0.046 & 0.017 \pm 0.044 & 0.001 \pm 0.000 & 

# 2E-1
0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & \
0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & 0.000 \pm 0.000 & \
0.297 \pm 0.106 & 0.226 \pm 0.139 & 0.240 \pm 0.136 & 0.045 \pm 0.111 & 0.014 \pm 0.065 & \
0.248 \pm 0.002 & 0.248 \pm 0.002 & 0.248 \pm 0.002 & 0.135 \pm 0.121 & 0.031 \pm 0.078 & \
0.500 \pm 0.000 & 0.500 \pm 0.000 & 0.500 \pm 0.000 & 0.296 \pm 0.244 & 0.033 \pm 0.118 & \
0.148 \pm 0.032 & 0.148 \pm 0.033 & 0.152 \pm 0.025 & 0.014 \pm 0.042 & 0.002 \pm 0.000 &
'''

## Plot final error

T10 = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray','olive', 'cyan']
str_surface = ['(a) Saddle','(b) Rot-saddle','(c) Seesaw','(d) Monk-saddle','(e) Anti-saddle','(f) Weapons']
#['Saddle','Rot-saddle','Seesaw','Monk-saddle','Anti-saddle','Weapons']
str_method = ['GD','Alt GD','K=1','K=2','K=5','K=10']

plt.figure(2)
plt.clf()
plt.subplot(4,1,1)

w = 1./9
s = 1./6
for k in range(1,nmethods):
    pos = (k-(nmethods-1)/2.)*s
    plt.bar(np.arange(6)+pos, tmean[:,k], w, color=T10[k], label=str_method[k])
    for j in range(6):
        plt.errorbar(j+pos, tmean[j,k], 1.1*tstd[j,k], color=T10[k], lw=2, capsize=8, capthick=3)
        plt.text(j+pos,tmean[j,k]+1.1*tstd[j,k]+0.1,r'%3.2f'%(tmean[j,k]),horizontalalignment='center',size=10)
        plt.text(j+pos,tmean[j,k]+1.1*tstd[j,k]+0.05,r'$\pm%3.2f$'%(tstd[j,k]),horizontalalignment='center',size=10)
    #plt.errorbar(np.arange(6)+pos+w/2., tmean[:,k], np.vstack((.001*np.ones((6)),1.*tstd[:,k])), color=T10[k], fmt=None, lw=3, capsize=10, capthick=3)

plt.axis([-0.4,5.5,0.0,0.8])
#plt.ylabel('Error', fontsize=18)
plt.title('Error (distance to optima)', fontsize=16)
plt.xticks(np.arange(6), str_surface, size=16)
plt.legend(fontsize=12,loc='upper left')

plt.show(block=False)

raw_input('Press enter to continue')        
 
#with matplotlib.backends.backend_pdf.PdfPages(result_dir+'/errs_final_simple.pdf') as pdf:
#    pdf.savefig(bbox_inches='tight')




## Show individual trials

T = 200
ts = np.arange(T)
plt.figure(1)

for i in range(6):
    for j in range(1,nmethods):
        ax = plt.subplot(6,5,i*5+j)
        if i==0: # top_row
            ax.set_title(str_method[j],size=16)
        y = errs_norm[i,:,:T,j].squeeze().transpose((1,0))
        tmean = y.mean(1).squeeze()
        tstd = y.std(1).squeeze()
        # each trial
        plt.plot(ts,y,'y',alpha=0.5)
        # std
        plt.fill_between(ts,tmean-1.*tstd,tmean+1.*tstd,where=None,interpolate=True,color='c',alpha=0.5)#,facecolor = )
        # mean
        plt.plot(ts,tmean,'b',linewidth=2.)
        plt.axis((0,T,0,0.6))
        if j==1: # left
            plt.ylabel(str_surface[i],size=14)

plt.show(block=False)#False)

raw_input('Press enter to continue')        

#with matplotlib.backends.backend_pdf.PdfPages(result_dir+'/errs_simple.pdf') as pdf:
#    pdf.savefig(bbox_inches='tight')



## Frequency of convergence
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib import colors as mcolors
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#import tensorflow as tf
#from minimax_examples import *
    
max_iter = 1000
labels=['GD','Alt-GD','minimax1','minimax2','minimax5']
specs=['c-d','m-^','b-v','g-s','r-o']
convrates = np.load(os.path.join(result_dir,'convrates.npy')) # 6 thresholds x 6 surfaces x 5 methods    
'''

for i in range(6): # surfaces
    plt.figure(4)
    plt.clf()
    plt.subplot(2,1,1)
    #width = 1./nmethods
    #width = 1./8
    for k in range(nmethods): # methods
        #plt.bar(np.arange(3)-width*(5./2.-np.float(k)),convrates[3:6,i,k].squeeze(),0.9*width,color=specs[k][0],label=labels[k])
        plt.bar(np.arange(3)+(np.float(k)-nmethods/2.)/(nmethods+3),convrates[1:,i,k].squeeze(),0.85/(nmethods+3),color=specs[k][0],label=labels[k])        
    
    #plt.bar(np.arange(3),convrates[4,i,:].squeeze(),width,color=specs[i][0],label=labels[i])
    #plt.bar(np.arange(3)+width,convrates[5,i,:].squeeze(),width,color=specs[i][0],label=labels[i])
    plt.xticks(np.arange(3),['th=0.05','th=0.02','th=0.01'],fontsize=12)
    plt.yticks(np.arange(0.,1.2,.2),fontsize=12)
    #plt.xticks(range(max_iter), ['Saddle','Rot-saddle','Seesaw','Monk-saddle','Anti-saddle','Weapons'])#rotation='vertical')
    #plt.xlabel('iter',fontsize=12)
    #plt.ylabel('gradient norm',fontsize=12)
    #plt.axis([-0.1, 4.1, 0.5, 1.])
    #plt.legend(fontsize=18, loc='upper right')
    #plt.legend(fontsize=12, loc='upper center',ncol=3)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode='expand', borderaxespad=0.,fontsize=12)    
    plt.show(block=False)#False)
    raw_input('Press enter to continue')        
    #with matplotlib.backends.backend_pdf.PdfPages(os.path.join(result_dir,'convergence_simple_%d.pdf'%(i))) as pdf:
    #    pdf.savefig(bbox_inches='tight')






