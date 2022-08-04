#!/usr/bin/env python

# std libs
import re
import numpy as np
import scipy.stats as stats
import modules.basic as basic
import modules.geo as geo
import modules.data as data
import modules.correlation as corr

from sklearn.kernel_ridge import KernelRidge as KR
import numpy.linalg as la

###############################################################################
# classes
###############################################################################
class NadineGoris_base(data.field):
    def __init__(self):
        super(NadineGoris_base,self).__init__(['all'])

        # read Qs
        lines=open('data/Y_axes.txt','r').read().split("\n")
        p1='[a-zA-Z][a-zA-Z\-]+\d*[a-zA-Z\-]*\d*'
        p2='\d+\.\d+'
        p3='1850-2099|2090s'
        pattern=p1+'|'+p2+'|'+p3
        store_1850=store_2090s=0
        q_1850={}
        q_2090={}
        for line in lines:
            n=re.findall(pattern,line)
            if not n: continue
            if n[0]=='1850-2099':
                store_1850=1
                store_2090s=0
                continue
            if n[0]=='2090s':
                store_1850=0
                store_2090s=1
                continue
            if store_1850: q_1850.update({n[0]:float(n[1])})
            if store_2090s: q_2090.update({n[0]:float(n[1])})
        self._q_1850_2099=np.array([q_1850[name] for name in self._names])
        self._q_2090=np.array([q_2090[name] for name in self._names])

    def QvsQ(self, M):
        m=M.mask()
        fam=self._fa[m,:]
        Q=np.sum(fam,axis=0)
        return(np.transpose(np.vstack((Q,self._Q))))

    def cost(self, m):
        pass

    def __str__(self):
        s=super(NadineGoris_base,self).__str__()
        return(s)

class NadineGoris_1(NadineGoris_base):
    def __init__(self, Q='2090s', models=[]):

        # initialize super
        super(NadineGoris_1,self).__init__()

        # initial cost function
        if not models: self._models=list(range(0,self.m()))
        else: self._models=models
        self._fa=(self._field*self._area)[:,self._models]
        self._a=self._area; self._a=self._a/np.sum(self._a)
        if Q=='2090s': self._Q=self._q_2090[self._models]; self._Qname='2090s'
        elif Q=='1850-2099': self._Q=self._q_1850_2099[self._models]; self._Qname='1850-2099'
        else: raise Exception("NadineGoris_1: Initialization error, Q '%s' not valid" % Q)
        self._nlcorr=0
        self._nlcorr_h=1.0

    def nlcorr(self, nlcorr, h=1.0):
        self._nlcorr=nlcorr
        self._nlcorr_h=h

    def cost(self, M):
        m=M.mask()
        if np.sum(m)==0: return(-2,0)
        fam=self._fa[m,:]
        am=np.sum(self._a[m,:])
        Q=np.sum(fam,axis=0)
        if not self._nlcorr:
            c=stats.pearsonr(Q,self._Q)
            return(c[0],am)
        elif self._nlcorr==1:
            c=basic.nlcorr(Q,self._Q,h=self._nlcorr_h)
            return(c,am)
        else:
            c=corr.dcorr(Q,self._Q)
            return(c,am)

    def __str__(self):
        s="--- NadineGoris 1 ---\n"
        if not self._nlcorr: s=s+"Cost: Pearson\n"
        else: s=s+"Cost: NL correlation\n"
        s=s+"Q: %s\n" % self._Qname
        s=s+"Models: %s\n\n" % str(self._models)
        s=s+super(NadineGoris_1,self).__str__()
        return(s)

def kr(X,Y):
    X=X-np.min(X); X=X/np.max(X); X=X.reshape(-1,1)
    Y=Y-np.min(X); Y=Y/np.max(Y); Y=Y.reshape(-1,1)
    kr=KR(alpha=0.001, kernel='rbf', gamma=1)
    kr.fit(X,Y)
    Z=kr.predict(X)
    ey=la.norm(Y-Z)/(la.norm(Y)+la.norm(Z))
    kr.fit(Y,X)
    Z=kr.predict(Y)
    ex=la.norm(X-Z)/(la.norm(X)+la.norm(Z))
    e=min(ex,ey)
    return(1-e)










