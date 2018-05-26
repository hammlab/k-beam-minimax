## minimax_examples.py
'''
Various 2D examples surfaces for testing minimax algorithms.

Examples of saddle points = f1, f2, f8
Examples of no saddle points = f32, f12, f9
'''

import os
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib import colors as mcolors
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


######################################################################################################

# Saddle (quadratic, separable): 0.5(u^2-v^2)
def f1(u,v,args):
    # maxmin: (0,0)
    # minmax: (0,0)
    # critical point: (0,0)
    # saddle points: (0,0)
    # m=2, M=2
    f = 5.*(u**2 - v**2)
    fu = 5.*2.*u #dZ/dU = 2U = 0
    fv = -5.*2.*v #dZ/dV = -2V = 0
    fuu = 5.*2.
    fvv = -5.*2.
    fuv = 0.
    fvu = fuv
    #min_x max_y Z = max_y min_x Z = Z(0,0)
    return [f,fu,fv,fuu,fuv,fvu,fvv]

# Rotated saddle (quadratic): u^2-v^2+2uv
def f2(u,v,args):
    # maxmin: (0,0)
    # minmax: (0,0)
    # critical point: (0,0)    
    # saddle points: (0,0)
    # m=2, M=2
    f = u**2 - v**2 + 2.*u*v
    fu = 2.*u + 2.*v#dZ/dU = 2U = 0
    fv = -2.*v + 2.*u#dZ/dV = -2V = 0
    fuu = 2.
    fvv = -2.
    fuv = 2.
    fvu = fuv
    #min_x max_y Z = max_y min_x Z = Z(0,0)
    return [f,fu,fv,fuu,fuv,fvu,fvv]


# Monkey saddle (3rd order)
def f3(u,v,args):
    # maxmin: (-0.4,+-0.2), (0.2,+-0.2)?
    # minmax: (0, all?)
    # critical point: (0,0)
    # No saddle points
    # m=0, M=6*max(max|u|,max|v|)
    f = u**3 - 3.*u*(v**2) 
    fu = 3.*(u**2)-3.*(v**2)
    fv = -6.*u*v
    fuu = 6.*u
    fvv = -6.*u
    fuv = -6.*v
    fvu = fuv
    #d2Z/dU2 = 6U
    #dZ/dV = -6UV = 0
    #dZ/dU = 3U^2 - 3V^2 = 3(x-y)(x+y)=0
    #d2Z/dU2 = 6U
    #dZ/dV = -6UV = 0
    #d2Z/dV2 = -6U
    return [f,fu,fv,fuu,fuv,fvu,fvv]


# Rotated Monkey saddle: v^3 - 3vu^2
def f32(u,v,args):
    # maxmin: (all,0)
    # minmax: (-0.25,-0.2), (+0.25,-0.2)?
    # critical point: (0,0)
    # local saddle point: none. (0,0) is not a local maximum of v...
    # No saddle points except at boundaries?
    # m=0, M=6*max(max|u|,max|v|)
    f = v**3 - 3.*v*(u**2) 
    fu = -6.*u*v
    fv = 3.*(v**2)-3.*(u**2)
    fuu = -6.*v
    fvv = 6.*v
    fuv = -6.*u
    fvu = fuv
    return [f,fu,fv,fuu,fuv,fvu,fvv]



# Another (non-polynomial) saddle 
# f(u,v) = u'Av+ b'u+c'v - log(1-u'u) + log(1-v'v)
# constrained on u'u <1 and v'v < 1
def f4(u,v,args): # u,v: (D,) array
    # 
    A,b,c = args
    du,dv = A.shape
    uu = (u**2).sum()
    vv = (v**2).sum()
    f = np.dot(u,np.dot(A,v))+np.dot(b,u)+np.dot(c,v)-np.log(1.-uu)+np.log(1.-vv)
    fu = np.dot(A,v)+b+2./(1.-uu)*u
    fuv = A#.T
    fvu = np.transpose(fuv)
    fuu = 2./(1.-uu)*np.eye(du)+4./((1.-uu)**2)*np.outer(u,u)
    fv = np.dot(A.T,u)+c-2./(1.-vv)*v
    fvv = -2./(1.-vv)*np.eye(dv)-4./((1.-vv)**2)*np.outer(v,v)
    
    return [f,fu,fv,fuu,fuv,fvu,fvv]


# sin-cos
def f5(u,v,args):
    # minmax = (1/(2a), all) and (-1/(2a),all)
    # maxmin = (all, 0)
    # critical point = (0,-1/2b), (0,1/2b), (1/2a,0), (-1/2a,0)
    # saddle points: (1/2a,0), (-1/2a,0)
    # m=0, M=2*c*max(a,b)
    a = 1.5
    b = 1.5
    c = .1
    f = -c*np.cos(a*np.pi*u)*np.sin(b*np.pi*v)
    fu = c*a*np.pi*np.sin(a*np.pi*u)*np.sin(b*np.pi*v)
    fv = -c*b*np.pi*np.cos(a*np.pi*u)*np.cos(b*np.pi*v)
    fuu = c*(a*np.pi)**2*np.cos(a*np.pi*u)*np.sin(b*np.pi*v)
    fvv = c*(b*np.pi)**2*np.cos(a*np.pi*u)*np.sin(b*np.pi*v)
    fuv = c*a*np.pi*b*np.pi*np.sin(a*np.pi*u)*np.cos(b*np.pi*v)
    fvu = fuv

    return [f,fu,fv,fuu,fuv,fvu,fvv]


# Simple function with a single 2nd order term
# 2  3 / 0 1
def f6(u,v,args):
    # minmax: (-0.6,-0.6) boundary
    # maxmin: (-0.6,-0.6) boundary
    # critical points: none
    # saddle points:(-0.6,-0.6), boundary
    # m=0, M=max|a|
    #solve: (u,v): (-1,-1), (1,-1), (1,1), (-1,1)
    A=np.asarray([[-1.*-1,-1.,-1., 1.],[1.*-1,1.,-1., 1.],[1.*1,1.,1., 1.],[-1.*1,-1.,1., 1.]])
    abcd = solve(A,np.asarray([2.,3.,1.,0.]))
    #abcd = solve(A,np.asarray([3.,1.,2.,0.]))
    a = abcd[0]
    b = abcd[1]
    c = abcd[2]
    d = abcd[3]            
    f = a*u*v + b*u + c*v + d
    fu = a*v + b
    fv = a*u + c
    fuu = 0.
    fvv = 0.
    fuv = a
    fvu = fuv
    return [f,fu,fv,fuu,fuv,fvu,fvv]


# Simple function with a single 2nd order term
def f7(u,v,args):
    # minmax: (0.1,all), 
    # maxmin: (all, 0.1), 
    # critical points: (0.1,0.1)
    # minimax points: (0.1,0.1), I think 
    # m=0, M=max|a|
    #
    #solve: (u,v): (-1,-1), (1,-1), (1,1), (-1,1)
    A=np.asarray([[-1.*-1,-1.,-1., 1.],[1.*-1,1.,-1., 1.],[1.*1,1.,1., 1.],[-1.*1,-1.,1., 1.]])
    #abcd = solve(A,np.asarray([2.,3.,1.,0.]))
    abcd = solve(A,np.asarray([3.,0.,2.,0.]))
    a = abcd[0]
    b = abcd[1]
    c = abcd[2]
    d = abcd[3]            
    f = a*u*v + b*u + c*v + d
    fu = a*v + b
    fv = a*u + c
    fuu = 0.
    fvv = 0.
    fuv = a
    fvu = fuv
    return [f,fu,fv,fuu,fuv,fvu,fvv]


# Seesaw: -v*sin(pi*u)
#\[
#f(u,v) = -v\sin(pi*u),\;\; |u| \leq 1/2,\;\; |y|\leq 1/2.
#\]
# fv = 0: u = ..,-1,0,1,...
# fu = 0: v = 0 or u= +-1/2, +-3/2, +-5/2,...
# critical (0,0)
# saddle (0,0)
# minimax: (0,[-.5,.5])

def f8(u,v,args):
    #
    f = -v*np.sin(np.pi*u)
    fu = -v*np.pi*np.cos(np.pi*u)
    fv = -np.sin(np.pi*u)
    fuu = v*((np.pi)**2)*np.sin(np.pi*u)
    fuv = -np.pi*np.cos(np.pi*u)
    fvu = fuv
    fvv = 0.
    return [f,fu,fv,fuu,fuv,fvu,fvv]


   
# Weapons allocation problem.
#f(u,v) = -\sum_i v_i(1-e^{-\beta_i (x_i+.5) e^{-\alpha_i (y_i+.5)}}),
# \sum_i (x_i+.5) = X,\;\sum_i (y_i+.5) = Y, \;|x_i|\leq.5, |y_i|\leq.5,\;v_i, \alpha_i,\beta_i>0.
# Special case: after translation
# n=2, v1=v2=al1=al2=X=Y=1, beta1=beta2=10
# f(u,v) = -\sum_{i=1}^2 (1-e^{-10(x_i+.5) e^{-(y_i+.5)}}),\;x_1+x_2=0,\;y_1+y_2=0.
# $\min\max$-solution: -1.834 at $(x_1,y_1)=(0.,-.5)$ and $(x_1,y_1)=(0.,.5)$.
# $\max\min$-solution: -1.88 at $(.158, -.5)$ and $(.158,.5)$.
# 
def f9(u,v,args):
    f = -(1.-np.exp(-10.*(u+.5)*np.exp(-(v+.5))) + 1.-np.exp(-10.*(1.-(u+.5))*np.exp(-1.+(v+.5))))
    fu = -(10.*np.exp(-(v+.5))*np.exp(-10.*(u+.5)*np.exp(-(v+.5))) + -10.*np.exp(-1.+(v+.5))*np.exp(-10.*(1.-(u+.5))*np.exp(-1.+(v+.5))))
    fv = -(-np.exp(-10.*(u+.5)*np.exp(-(v+.5)))*(-10.*(u+.5)*np.exp(-(v+.5))*-1.) + -np.exp(-10.*(1.-(u+.5))*np.exp(-1.+(v+.5)))*(-10.*(1.-(u+.5))*np.exp(-1.+(v+.5))))
    u = np.array(u)
    v = np.array(v)
    fuu = np.nan*np.ones((u.size,u.size))
    fuv = -(-10.*np.exp(-(v+.5))*np.exp(-10.*(u+.5)*np.exp(-(v+.5)))
            +10.*np.exp(-(v+.5))*np.exp(-10.*(u+.5)*np.exp(-(v+.5)))*(+10.)*(u+.5)*np.exp(-(v+.5))
            -10.*np.exp(-1.+(v+.5))*np.exp(-10.*(1.-(u+.5))*np.exp(-1.+(v+.5)))
            -10.*np.exp(-1.+(v+.5))*np.exp(-10.*(1.-(u+.5))*np.exp(-1.+(v+.5))))
    #fuv = np.nan*np.ones((u.size,v.size))
    fvu = np.nan*np.ones((v.size,u.size))
    fvv = np.nan*np.ones((v.size,v.size))
                            
    return [f,fu,fv,fuu,fuv,fvu,fvv]


# Demyanov
# original f(u,v) = cos(u*(v+pi/4)),  0<=v<=3*pi/2,  u
# Translate:  [pi/4,7*pi/4] => cos(pi*u*(3/2*(v+1/2)+1/4))=cos(pi*u*(1.5*v+1.)),  |v|<=1/2  =>  [pi/4,7*pi/4]
# scale u by a
def f10(u,v,args):
    a = 3.
    f = 1./a*np.cos(a*np.pi*u*(1.5*v+1.))
    fu = -1./a*np.sin(a*np.pi*u*(1.5*v+1.))*a*np.pi*(1.5*v+1.)
    fv = -1./a*np.sin(a*np.pi*u*(1.5*v+1.))*a*np.pi*1.5*u
    fuu = np.nan*np.ones((u.size,u.size))
    fuv = np.nan*np.ones((u.size,v.size))
    fvu = np.nan*np.ones((v.size,u.size))
    fvv = np.nan*np.ones((v.size,v.size))
    return [f,fu,fv,fuu,fuv,fvu,fvv]


# Counter example
def f11(u,v,args):
    f = v**2-np.abs(u-v)
    fu = -np.sign(u-v)
    fv = 2.*v-np.sign(v-u)
    fuu = np.nan*np.ones((u.size,u.size))
    fuv = np.nan*np.ones((u.size,v.size))
    fvu = np.nan*np.ones((v.size,u.size))
    fvv = np.nan*np.ones((v.size,v.size))
    return [f,fu,fv,fuu,fuv,fvu,fvv]


# Anti-saddle: v^2+2uv-u^2
# critical point: (0,0)
# saddle point: none
# local saddle point: none
# minimax point: (0,+-0.5)
def f12(u,v,args):
    f = 2.*v**2-(u-v)**2
    fu = -2.*(u-v)
    fv = 4.*v-2.*(v-u)
    u = np.array(u)
    v = np.array(v)
    fuu = -2.*np.ones((u.size,u.size))
    fuv = +2.*np.ones((u.size,v.size))
    fvu = +2.*np.ones((v.size,u.size))
    fvv = +2.*np.ones((v.size,v.size))
    return [f,fu,fv,fuu,fuv,fvu,fvv]


# -0.5u^2 +2uv -v^2
# Anti-saddle: v^2+2uv-u^2
# critical point: (0,0)
# saddle point: none
# local saddle point: none
# minimax point: (0,+-0.5)
def f13(u,v,args):
    f = -0.5*(u**2) + 2.*u*v - v**2
    fu = -1.*u + 2.*v
    fv = 2.*u -2.*v
    u = np.array(u)
    v = np.array(v)
    fuu = -1.*np.ones((u.size,u.size))
    fuv = +2.*np.ones((u.size,v.size))
    fvu = +2.*np.ones((v.size,u.size))
    fvv = -2.*np.ones((v.size,v.size))
    return [f,fu,fv,fuu,fuv,fvu,fvv]


#################################################################################################

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)



if __name__ == "__main__":
    # Examples of saddle points = f1, f2, f8
    # Examples of no saddle points = f32, f12, f9

    fn = f13
    args = np.zeros(0)
    
    ## Make surfce
    u = np.linspace(-0.5,0.5,100)
    v = np.linspace(-0.5,0.5,100)    
    
    #U,V = np.meshgrid(np.linspace(-2,2,100),np.linspace(-2,2,100))
    U,V = np.meshgrid(u,v)
    Z = np.zeros(U.shape)
    Fu = np.zeros(U.shape)
    Fv = np.zeros(U.shape)    
    
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            #U[i,j],V[i,j] = proj_linf(U[i,j],V[i,j])
            Z[i,j],Fu[i,j],Fv[i,j],_,_,_,_ = fn(U[i,j],V[i,j],args)

    if True: ## surface
        fig = plt.figure(1)
        ax = fig.gca(projection='3d')
        if True:
            ax.plot_surface(U[:,:],V[:,:],Z[:,:],\
                cmap=cm.coolwarm,rstride=5,cstride=5,edgecolors='none',antialiased=True)
        else:
            ax.plot_wireframe(U,V,Z,rstride=5,cstride=5,linewidth=0.5,color='k')#gray')
            
        if True: ## Add saddle and minimax points
            if False: ## f1, f2, f8
                ax.plot([0.],[0.],[fn(0.,0.,args)[0]],'o',markersize=6,markeredgewidth=1,markeredgecolor='k',markerfacecolor='k')
                ax.plot([0.],[0.],[fn(0.,0.,args)[0]],'o',markersize=15,markeredgewidth=1,markeredgecolor='k',markerfacecolor='None')
            if False: ## f32
                ax.plot([-0.25],[-0.25],[fn(-0.25,-0.25,args)[0]],'o',markersize=6,markeredgewidth=1,markeredgecolor='k',markerfacecolor='k')
                ax.plot([+0.25],[-0.25],[fn(+0.25,-0.25,args)[0]],'o',markersize=6,markeredgewidth=1,markeredgecolor='k',markerfacecolor='k')
                ax.plot([-0.25],[0.5],[fn(-0.25,0.5,args)[0]],'o',markersize=6,markeredgewidth=1,markeredgecolor='k',markerfacecolor='k')
                ax.plot([+0.25],[0.5],[fn(+0.25,0.5,args)[0]],'o',markersize=6,markeredgewidth=1,markeredgecolor='k',markerfacecolor='k')
            if False: ## f12
                ax.plot([0.],[+0.5],[fn(0.,+0.5,args)[0]],'o',markersize=6,markeredgewidth=1,markeredgecolor='k',markerfacecolor='k')
                ax.plot([0.],[-0.5],[fn(0.,-0.5,args)[0]],'o',markersize=6,markeredgewidth=1,markeredgecolor='k',markerfacecolor='k')
            if False: ## f9
                ax.plot([0.],[+0.5],[fn(0.,+0.5,args)[0]],'o',markersize=6,markeredgewidth=1,markeredgecolor='k',markerfacecolor='k')
                ax.plot([0.],[-0.5],[fn(0.,-0.5,args)[0]],'o',markersize=6,markeredgewidth=1,markeredgecolor='k',markerfacecolor='k')
            if True: ## f13
                ax.plot([0.],[0.],[fn(0.,+0.,args)[0]],'o',markersize=6,markeredgewidth=1,markeredgecolor='k',markerfacecolor='k')
                #ax.plot([0.],[-0.5],[fn(0.,-0.5,args)[0]],'o',markersize=6,markeredgewidth=1,markeredgecolor='k',markerfacecolor='k')
        
        
        if False: ## Add gradient
            s = 15.#
            du = -s*Fu/np.sqrt((Fu**2+Fv**2).sum())
            dv = s*Fv/np.sqrt((Fu**2+Fv**2).sum())
            tZ = np.zeros(U.shape)
            for i in range(U.shape[0]):
                for j in range(U.shape[1]):
                    tZ[i,j] = fn(U[i,j]+du[i,j],V[i,j]+dv[i,j],args)[0]
            dz = tZ - Z
            if True: ## Normalize
                tsum = np.sqrt((du**2 + dv**2 + dz**2).sum())
                du = s*du/tsum
                dv = s*dv/tsum
                dz = s*dz/tsum
                #print np.sqrt(du**2 + dv**2 + dz**2)/s
            for i in range(0,U.shape[0],10):
                for j in range(0,U.shape[1],10):
                    #ax.arrow(U[i,j],V[i,j],-s*Fu[i,j],s*Fv[i,j],width=0.001,length_includes_head=True)
                    a = Arrow3D([U[i,j],U[i,j]+du[i,j]], [V[i,j],V[i,j]+dv[i,j]], [Z[i,j],Z[i,j]+dz[i,j]], mutation_scale=8,lw=1,arrowstyle='-|>',color='k')
                    ax.add_artist(a)
            #ax.quiver(U[:,:,V[:,:]

        ax.view_init(elev=60, azim=-45) #Reproduce view
        ax.set_xlabel('U',fontsize=18)
        ax.set_ylabel('V',fontsize=18)
        #plt.legend(fontsize=18, loc='upper right')
        plt.show(block=False)
        ## Change file name
        #with matplotlib.backends.backend_pdf.PdfPages(os.path.join(result_dir,'demo_surfaces9_points.pdf')) as pdf:
        #    pdf.savefig(bbox_inches='tight')


    if True:
        ## Show max functions
        fig = plt.figure(2)
        phi = Z.max(0)
        plt.plot(u,phi)
        plt.xlabel('U',fontsize=18)
        plt.show(block=False)
        #with matplotlib.backends.backend_pdf.PdfPages(os.path.join(result_dir,'results/maxfunc12.pdf')) as pdf:
        #    pdf.savefig(bbox_inches='tight')

    if True:
        ## Show min functions
        fig = plt.figure(3)
        phi_ = Z.min(1)
        plt.plot(v,phi_)
        plt.xlabel('V',fontsize=18)
        plt.show(block=False)
        #with matplotlib.backends.backend_pdf.PdfPages(os.path.join(result_dir,'maxfunc12.pdf')) as pdf:
        #    pdf.savefig(bbox_inches='tight')

    if True: ## Show global saddle points, if any
        fig = plt.figure(4)
        #plt.close()
        #fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_wireframe(U,V,Z,rstride=5,cstride=5,linewidth=0.5,color='k')#gray')
        ## argmin_u f(u,v) 
        Zmin = np.min(Z,1)
        eps = 1E-4
        for i in range(100): # for each v
            ind = np.where(Z[i,:]<=Zmin[i]+eps)[0]
            for j in ind:
                ax.scatter(u[j],v[i],Z[i,j],c='r',alpha=0.5)
        ## argmax_v f(u,v) 
        Zmax = np.max(Z,0)
        eps = 1E-4
        for i in range(100): # for each u
            ind = np.where(Z[:,i]>=Zmax[i]-eps)[0]
            for j in ind:
                ax.scatter(u[i],v[j],Z[j,i],c='b',alpha=0.5)

        ax.view_init(elev=60, azim=-45) #Reproduce view
        ax.set_xlabel('U',fontsize=18)
        ax.set_ylabel('V',fontsize=18)
        plt.legend(fontsize=18, loc='upper right')
        plt.show(block=False)

    if True: ## Show cricial points, if any
        fig = plt.figure(5)
        #plt.close()
        #fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_wireframe(U,V,Z,rstride=5,cstride=5,linewidth=0.5,color='k')#gray')
        ## Fu=0
        eps = 1E-2
        for i in range(100): # for each v
            ind = np.where(np.abs(Fu[i,:])<=eps)[0]
            for j in ind:
                ax.scatter(u[j],v[i],Z[i,j],c='r',alpha=0.5)
        ## Fv=0
        eps = 1E-2
        for i in range(100): # for each u
            ind = np.where(np.abs(Fv[:,i])<=eps)[0]
            for j in ind:
                ax.scatter(u[i],v[j],Z[j,i],c='b',alpha=0.5)

        ax.view_init(elev=60, azim=-45) #Reproduce view
        ax.set_xlabel('U',fontsize=18)
        ax.set_ylabel('V',fontsize=18)
        plt.legend(fontsize=18, loc='upper right')
        plt.show(block=False)
    
    raw_input('Press any key to continue')

