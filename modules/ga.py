#!/usr/bin/env python

# std libs
import pandas as pd
import numpy as np
import random
import numpy.linalg as la
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import modules.problem as problem
import modules.data as data
import time
import copy
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from sklearn.kernel_ridge import KernelRidge as KR
from scipy.spatial.distance import cdist

###############################################################################
# GA_simple_float
###############################################################################
class GA_simple_float(problem.NadineGoris_1):
    def __init__(self, area=[], gene=0, pop=0, ga_eval='', hof=1, connected=0, models=[], verbose=1):
        super(GA_simple_float,self).__init__(Q='2090s', models=models)

        self._connected = connected
        self._ga_eval = ga_eval
        self._arealim = area
        self._pop_n = pop
        self._GENE = gene
        self._hof_n = hof
        self._CXPB = 0.5
        self._MUTPB = 0.2
        self._m = np.full(self.n(), False, dtype=bool)
        self._M = data.mask(['m',self,self._m])
        self._neval = 0;
        self._best = []
        self.verbose = verbose

        self._pn=np.copy(self._points)
        self._pn=self._pn-np.min(self._pn,axis=0)
        self._pn=self._pn/np.max(np.max(self._pn,axis=0))
        self._pn_max=np.max(self._pn,axis=0)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        self._toolbox=base.Toolbox()
        self._toolbox.register("attr_float", random.random)
        self._toolbox.register("individual", tools.initRepeat, creator.Individual, self._toolbox.attr_float, self._GENE)
        self._toolbox.register("population", tools.initRepeat, list, self._toolbox.individual)
        self._toolbox.register("mate", tools.cxOnePoint)
        self._toolbox.register("select", tools.selTournament, tournsize=10)
        self._toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.2)
        self._toolbox.register("evaluate", self._ga_eval, points=self._pn, cost_fct=self.cost, M=self._M, arealim=self._arealim, connected=self._connected, xymax=self._pn_max)
        self._hof=tools.HallOfFame(self._hof_n)

        self._history=tools.History()
        self._toolbox.decorate("mate", self._history.decorator)
        self._toolbox.decorate("mutate", self._history.decorator)

        self._pop=self._toolbox.population(n=self._pop_n)
        self._history.update(self._pop)
        fitnesses=list(map(self._toolbox.evaluate, self._pop))
        self._best+=[max(fitnesses)[0]]
        for ind, fit in zip(self._pop, fitnesses):
            ind.fitness.values = fit
        self._neval+=len(fitnesses)

    def iter(self, n=100):
        if self.verbose:
            print("--- GA iter (n=%d) ---" % n)
        for g in range(0,n):
            offspring = self._toolbox.select(self._pop, self._pop_n)
            offspring = list(map(self._toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self._CXPB:
                    self._toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self._MUTPB:
                    self._toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            self._neval+=len(invalid_ind)
            fitnesses = list(map(self._toolbox.evaluate, invalid_ind))
            self._best+=[max(max(self._best),max(max(fitnesses)))]
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            self._pop[:] = offspring
            fits = [ind.fitness.values[0] for ind in self._pop]
            self._hof.update(self._pop)
            if self.verbose:
                print(f'iter {g + 1:2d}: max = {np.max(fits):.3f}, mean = {np.mean(fits):.3f}, min = {np.min(fits):.3f}')
        if self.verbose:
            print()

    def hof(self):
        items=self._hof.items
        ret=[]
        for item in items:
            M=self._ga_eval(item, self._pn, 0, self._M, self._arealim, self._connected, self._pn_max)
            ret=ret+[copy.deepcopy(M)]
        return(ret)

    def gof(self):
        items=self._hof.items
        ret=[]
        for item in items:
            a=np.array(copy.deepcopy(item))
            ret=ret+[a]
        return(ret)

    def neval(self):
        return(self._neval)

    def best(self):
        return(np.array(self._best).reshape(-1))

    def plot(self):
        graph = nx.DiGraph(self._history.genealogy_tree)
        graph = graph.reverse()
        colors = [self._toolbox.evaluate(self._history.genealogy_history[i])[0] for i in graph]
        positions = graphviz_layout(graph, prog='dot')
        nx.draw(graph, positions, node_color=colors)
        plt.show()

    def item2str(self, item):
        return('')

    def res(self):
        df = pd.DataFrame({'correlation': pd.Series(dtype='float'), 'area': pd.Series(dtype='float'), 'n_points': pd.Series(dtype='int'), 'points': pd.Series(dtype='object')})
        items=self._hof.items
        for i, item in enumerate(items):
            M = self._ga_eval(item, self._pn, 0, self._M, self._arealim, self._connected, self._pn_max)
            c, a = self.cost(M)
            p = M.points()[M._A]
            n = p.shape[0]
            df = df.append({'correlation': c, 'area': a, 'n_points': n, 'points': p}, ignore_index = True)

        return(df)

    def __str__(self):
        s="--- Hall of fame ---\n"
        items=self._hof.items
        for i, item in enumerate(items):
            M = self._ga_eval(item, self._pn, 0, self._M, self._arealim, self._connected, self._pn_max)
            (c,a) = self.cost(M)
            s = s + f'nb {i}: pearson-corr = {c:.3f}, area = {a:.3f}{self.item2str(item)}\n'
        s=s+"\n"
        s=s+super(GA_simple_float,self).__str__()

        return(s)

###############################################################################
# Rectangle
###############################################################################
def Rectangle_eval(individual, points, cost_fct, M, arealim, connected, xymax):

    # set up
    nrec=int(len(individual)/4)
    penalty=0

    # eval rectangles
    for i in range(nrec):
        x1=individual[4*i]; x2=individual[4*i+1]; y1=individual[4*i+2]; y2=individual[4*i+3]
        if x1>x2: penalty+=x1-x2
        if y1>y2: penalty+=y1-y2
        if x1<0: penalty+=-x1
        if y1<0: penalty+=-y1
        if x2>xymax[0]: penalty+=x2-xymax[0]
        if y2>xymax[1]: penalty+=y2-xymax[1]
        m1=abs(points[:,0]-x1)+abs(points[:,0]-x2)<=abs(x1-x2)+0.001
        m2=abs(points[:,1]-y1)+abs(points[:,1]-y2)<=abs(y1-y2)+0.001
        if i: m=m+(m1*m2)
        else: m=m1*m2

    # eval if connected
    if connected:
        for i in range(nrec-1):
            x1=individual[4*i]; x2=individual[4*i+1]; y1=individual[4*i+2]; y2=individual[4*i+3]
            xp1=individual[4*i+4]; xp2=individual[4*i+5]; yp1=individual[4*i+6]; yp2=individual[4*i+7]
            if xp1>x2: penalty+=xp1-x2
            if yp1>y2: penalty+=yp1-y2
            if xp2<x1: penalty+=x1-xp2
            if yp2<y1: penalty+=y1-yp2

    # finish
    M.set_mask(m)
    if not cost_fct: return(M)
    else:
        (c,a)=cost_fct(M)
        if a<arealim[0]: penalty+=arealim[0]-a
        if a>arealim[1]: penalty+=a-arealim[1]
        if penalty: c=-1-penalty
        return(c),

class Rectangle(GA_simple_float):
    def __init__(self, area=[], nrec=1, pop=100, hof=5, connected=0, models=[], verbose = 0):
        super(Rectangle,self).__init__(area=area, gene=4*nrec, pop=pop, ga_eval=Rectangle_eval, hof=hof, connected=connected, models=models, verbose = verbose)
        self._nrec=nrec

    def item2str(self, item):
        s = ''
        for k in range(self._nrec):
            s += f', rectangle[{k}] = [LL = ({item[k + 0]:.3f}, {item[k + 2]:.3f}), UR = ({item[k + 1]:.3f}, {item[k + 3]:.3f})]' 
        return(s)

    def __str__(self):
        s = '--- Rectangle ---\n'
        s = s + f'Population size: {self._pop_n}\n'
        if self._nrec > 1:
            s = s + f'Number of rectangles: {self._nrec}\n'
            s = s + f'Connected: {self._connected}\n'
        s = s + f'Area: [{self._arealim[0]:.3f}, {self._arealim[1]:.3f}]\n\n'
        s = s + super(Rectangle,self).__str__()
        return(s)

###############################################################################
# Ellipse
###############################################################################
def Ellipse_eval(individual, points, cost_fct, M, arealim, connected, xymax):

    # set up
    if connected:
        nel=int((len(individual)-2)/3)
        fac=3
    else:
        nel=int(len(individual)/5)
        fac=5
    penalty=0

    # eval ellipses
    for i in range(nel):
        x1=np.array([individual[fac*i],individual[fac*i+1]])
        if x1[0]<0: penalty+=-x1[0]
        if x1[0]>xymax[0]: penalty+=x1[0]-xymax[0]
        if x1[1]<0: penalty+=-x1[1]
        if x1[1]>xymax[1]: penalty+=x1[1]-xymax[1]
        x2=np.array([individual[fac*i+3],individual[fac*i+4]])
        if x2[0]<0: penalty+=-x2[0]
        if x2[0]>xymax[0]: penalty+=x2[0]-xymax[0]
        if x2[1]<0: penalty+=-x2[1]
        if x2[1]>xymax[1]: penalty+=x2[1]-xymax[1]
        d=individual[fac*i+2]
        d12=la.norm(x1-x2)
        if d<d12: penalty+=d12-d
        d1=la.norm(points-x1,axis=1)
        d2=la.norm(points-x2,axis=1)
        m1=d1+d2<=d
        if i: m=m+m1
        else: m=m1

    M.set_mask(m)
    if not cost_fct: return(M)
    else:
        (c,a)=cost_fct(M)
        if a<arealim[0]: penalty+=arealim[0]-a
        if a>arealim[1]: penalty+=a-arealim[1]
        if penalty: c=-1-penalty
        return(c),

class Ellipse(GA_simple_float):
    def __init__(self, area=[], nel=1, pop=100, hof=5, connected=0, models=[], verbose = 0):
        if connected: 
            super(Ellipse,self).__init__(area=area, gene=2+3*nel, pop=pop, ga_eval=Ellipse_eval, hof=hof, connected=connected, models=models, verbose = verbose)
        else: 
            super(Ellipse,self).__init__(area=area, gene=5*nel, pop=pop, ga_eval=Ellipse_eval, hof=hof, connected=connected, models=models, verbose = verbose)
        self._nel=nel

    def item2str(self, item):
        s = ''
        if self._nel == 1:
            s = f', ellipse[0] = [F1 = ({item[0]:.3f}, {item[1]:.3f}), F2 = ({item[3]:.3f}, {item[4]:.3f}), d = {item[2]:.3f}]'
        return(s)

    def __str__(self):
        s = "--- Elipse ---\n"
        s = s + f'Population size: {self._pop_n}\n'
        if self._nel > 1:
            s = s + f'Number of ellipses: {self._nel}\n'
            s = s + f'Connected: {self._connected}\n'
        s = s + f'Area: [{self._arealim[0]:.3f}, {self._arealim[1]:.3f}]\n\n'
        s = s + super(Ellipse,self).__str__()
        return(s)






    


