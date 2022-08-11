# Genetic algorithm for regional optimisation of Emergent Constraints in the surface North Atlantic

This software is related to the COLUMBIA project and the 2d optimization in the publication 'Gulf Stream and interior western boundary volume transport as key regions to constrain the future North Atlantic Carbon Uptake' by Nadine Goris et al.

# COLUMBIA
COLUMBIA is an interdisciplinary project that aims to develop an innovative tool based on state-of-the-art machine learning technology, to efficiently analyze a large amount of model data to understand better why some models behave very differently than others. Combined with our current knowledge of how the climate system works based on the observational evidence, we will constrain the large spread in these model simulations. Also, using our novel tool, we hope to filter out those climate models that do not represent well the important dynamics observed in nature.

It is funded by Research Council of Norway (RCN) and led by **Research Professor Jerry Tjiputra** at NORCE - Norwegian Research Centre AS.

# Authors
The developer is **Chief Scientist Klaus Johannsen**, also at NORCE.

# Installation

```
git clone git@gitlab.norceresearch.no:columbia/GA_North_Atlantic_2d.git
cd GA_North_Atlantic_2d
```

# Usage
```
python3 run.py
```

# Modification
- Edit the configuration section in ```run.py```
```
N_RUNS = 10
POPULATION = 100
GENERATIONS = 10
AREA = [0.1,0.2]
MODELS = [0,1,2,3,4,5,6,7,8,9,10]
#Ga = ga.Rectangle
Ga = ga.Ellipse
```

```N_RUNS``` is the number of independent executions. As the algorithm is non-deterministic different runs may lead to different solutions. ```POPULATION``` is the size of the population in the genetic algorithm, ```GENERATIONS``` is the number of generations executed by the genetic algorithm. The ```AREA``` values determine the minimum resp. the maximum fraction of the physical area associated with the solution relative to the total area. Solutions with areas outside the specified interval will be penalized. ```MODELS``` is the list of models' indices to be considered which correspond to the models in ```modules/data.py```, line 15, given by names. The choice of ```Ga``` specifies the use of rectangles resp. ellipses.

# Results
The results of the ```N_RUNS``` independent executions are stored in ```results/```.


