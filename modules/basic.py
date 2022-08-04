#!/usr/bin/env python

# std libs
import warnings
import matplotlib.cbook
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#from matplotlib.mlab import griddata
from matplotlib.patches import Polygon
import pickle
from sklearn.kernel_ridge import KernelRidge as KR
import numpy.linalg as la

###############################################################################
# funtions
###############################################################################
def numpy_array_hash(a):
    b=a.reshape(-1)
    b=b[~np.isnan(b)]
    w=np.linspace(1,2,b.shape[0],endpoint=True).reshape(-1)
    return(hash(np.dot(b,w)))

def plot(x, xlabel='', ylabel='', title='', out='screen'):
    x=x[x[:,0].argsort(),:]
    plt.plot(x[:,0], x[:,1], '-o', color='blue');
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if out=='screen': plt.show()
    else: plt.savefig(out)
    plt.close()

def plot2(x, y, xlabel='', ylabel='', title='', out='screen'):
    x=x[x[:,0].argsort(),:]
    plt.plot(x[:,0], x[:,1], '-o', color='blue');
    plt.plot(y[:,0], y[:,1], '-', color='green');
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if out=='screen': plt.show()
    else: plt.savefig(out)
    plt.close()

def nlcorr(X,Y,h=1.0,N=0):
    # normalize X, Y
    X=X.reshape(-1,1); Xmean=np.mean(X); X=X-Xmean; Xnorm=la.norm(X); X=X/Xnorm;
    Y=Y.reshape(-1,1); Ymean=np.mean(Y); Y=Y-Ymean; Ynorm=la.norm(Y); Y=Y/Ynorm;

    # nonlinear correlation
    Z=np.hstack((X,Y))
    U,D,Vt=la.svd(Z,full_matrices=False)
    ZT=np.dot(Z,np.transpose(Vt))
    XT=ZT[:,0].reshape(-1,1)
    YT=ZT[:,1].reshape(-1,1)
    XTmin=np.min(XT)
    XTmax=np.max(XT)
    h=h*(XTmax-XTmin)
    kr=KR(alpha=0.001, kernel='rbf', gamma=1.0/h/h)
    kr.fit(XT,YT)
    fXT=kr.predict(XT)
    c=la.norm(YT-fXT)
    c=1-c*c

    if not N: return(c)

    # create array
    XT=np.linspace(XTmin,XTmax,N).reshape(-1,1);
    YT=kr.predict(XT)
    ZT=np.hstack((XT,YT))
    Z=np.dot(ZT,Vt)
    Z[:,0]=Z[:,0]*Xnorm+Xmean
    Z[:,1]=Z[:,1]*Ynorm+Ymean
    return(c,Z)

###############################################################################
# classes
###############################################################################
class pack:
    def __init__(self, object):
        self._type=type(object)
        self._pickle=pickle.dumps(object)

    def unpack(self):
        return(pickle.loads(self._pickle)) 

    def __str__(self):
        s="--- pack ---\n"
        s=s+"type:: %s\n" % self._type
        s=s+"size: %d\n" % len(self._pickle)
        return(s)


