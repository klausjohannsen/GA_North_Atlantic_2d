#!/usr/bin/env python

# std libs
import numpy as np
from sklearn.kernel_ridge import KernelRidge as KR
import numpy.linalg as la
import scipy.stats as stats

###############################################################################
# funtions
###############################################################################
def pearson(X,Y):
    corr=stats.pearsonr(X,Y)
    return(corr[0])


def svdkr(X,Y,h=1.0,N=0):
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

def dcorr(X,Y):
    n=len(X)

    ax=np.zeros((n,n))+X
    ax=np.abs(ax-np.transpose(ax))
    axcm=np.zeros((n,n))+np.mean(ax,axis=0)
    axrm=np.transpose(axcm)
    axm=np.zeros((n,n))+np.mean(ax)
    ax=ax-axcm-axrm+axm

    ay=np.zeros((n,n))+Y
    ay=np.abs(ay-np.transpose(ay))
    aycm=np.zeros((n,n))+np.mean(ay,axis=0)
    ayrm=np.transpose(aycm)
    aym=np.zeros((n,n))+np.mean(ay)
    ay=ay-aycm-ayrm+aym

    dcov=np.sqrt(np.mean(ax*ay))
    dvarx=np.sqrt(np.mean(ax*ax))
    dvary=np.sqrt(np.mean(ay*ay))

    return(dcov/np.sqrt(dvarx*dvary))

    






