#!/usr/bin/env python

# libs
import pandas as pd
import numpy as np
import pickle
import modules.basic as basic
import modules.ga as ga

########################################################################
# config
########################################################################
N_RUNS = 1000
POPULATION = 1000
GENERATIONS = 100
AREA = [0.1,0.2]
MODELS = [0,1,2,3,4,5,6,7,8,9,10]
Ga = ga.Rectangle
#Ga = ga.Ellipse

########################################################################
# run
########################################################################
print('config')
print(f'  N_RUNS: {N_RUNS}')
print(f'  POPULATION: {POPULATION}')
print(f'  GENERATIONS: {GENERATIONS}')
print(f'  AREA: {AREA}')
print(f'  MODELS: {MODELS}')
print(f'  Ga: {Ga.__name__}')
print()

# run
print('run')
B = np.zeros((N_RUNS, GENERATIONS + 1))
Best = {'M': None, 'Corr':-np.inf, 'Area': None}
for k in range(N_RUNS):
    print(f'  iteration {k}: ', end = '', flush = True)
    gen_alg = Ga(pop = POPULATION, area = AREA, hof = 1, models = MODELS)
    gen_alg.iter(n = GENERATIONS)
    B[k] = gen_alg.best()
    if B[k, GENERATIONS] > Best['Corr']:
        Best['Corr'] = B[k, GENERATIONS]
        M = gen_alg.hof()[0]
        Best['M'] = M.points()[M._A]
        Best['Area'] = gen_alg.cost(M)[1]
    print('done')
print()

# write convergence results to file
B_ens_min = np.min(B, axis=0)
B_ens_max = np.max(B, axis=0)
B_ens_mean = np.mean(B, axis=0)
B_ens_std = np.std(B, axis=0, ddof=1)
columns = ['B_ens_min','B_ens_max','B_ens_mean','B_ens_std']
X = np.transpose(np.vstack((B_ens_min, B_ens_max, B_ens_mean, B_ens_std)))
df = pd.DataFrame(X, columns = columns)
out = "results/convergence_%s_area=%.1f_%.1f_nrun=%d_pop=%d_iter=%d_exec=%d.txt" % (Ga.__name__.lower(), AREA[0], AREA[1], N_RUNS, POPULATION, GENERATIONS, 1)
df.to_csv(out, header=True, index=True, sep=' ', mode='a')

# write best shape to file
out = "results/data_best_%s_area=%.1f_%.1f_nrun=%d_pop=%d_iter=%d_exec=%d.txt" % (Ga.__name__.lower(), AREA[0], AREA[1], N_RUNS, POPULATION, GENERATIONS, 1)
meta = "which=best, corr=%1.3f, volume=%1.3f" % (Best['Corr'], Best['Area'])
np.savetxt(out, Best['M'], header = meta)









