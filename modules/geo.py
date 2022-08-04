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

###############################################################################
# functions
###############################################################################


###############################################################################
# classes
###############################################################################
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

class map:
    def __init__(self, lon_min_max=[-97.5, 19.5], lat_min_max=[0.5, 75.5], grid=[-90,10,10,70,20], title='',out='screen'):
        # center
        map_lon_0=np.mean(lon_min_max)
        map_lat_0=np.mean(lat_min_max)

        # basic paramters
        #plt.figure(figsize=(10,7))
        plt.figure(figsize=(8,5.6))
        self._title=title
        self._out=out

        # map
        self._map=Basemap(projection='gall', resolution='l', area_thresh=0.1, lat_0=map_lat_0, lon_0=map_lon_0, llcrnrlon=lon_min_max[0], llcrnrlat=lat_min_max[0], urcrnrlon=lon_min_max[1], urcrnrlat=lat_min_max[1])
        self._map.drawcoastlines()
        self._map.drawmapboundary(fill_color='aqua')
        self._map.fillcontinents(color='coral',lake_color='aqua')
        if len(grid)==5:
            n=(grid[3]-grid[2])/grid[4]+1
            parallels=np.linspace(grid[2],grid[3],n)
            self._map.drawparallels(parallels,labels=[False,True,True,False])
            n=(grid[1]-grid[0])/grid[4]+1
            meridians=np.linspace(grid[0],grid[1],n)
            self._map.drawmeridians(meridians,labels=[True,False,False,True])
        self._map.drawcountries()

    def plotmask(self,M):
        points=M.points()
        lons=points[:,0].tolist()
        lats=points[:,1].tolist()
        mask=M.mask().tolist()
        llm=list(zip(lons,lats,mask))

        cmap=matplotlib.cm.get_cmap('bwr')
        blue=cmap(0.0)
        red=cmap(1.0)
        black=cmap(0.5)

        for p in llm:
            x1,y1 = self._map(p[0]-0.5,p[1]-0.5)
            x2,y2 = self._map(p[0]+0.5,p[1]-0.5)
            x3,y3 = self._map(p[0]+0.5,p[1]+0.5)
            x4,y4 = self._map(p[0]-0.5,p[1]+0.5)
            if p[2]==0: color=blue
            elif p[2]==1: color=red
            else: color=black
            poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor=color)
            plt.gca().add_patch(poly)

    def plotfield(self,F,n,range=[0,0], mode='sym'):
        points=F.points()
        lons=points[:,0].tolist()
        lats=points[:,1].tolist()
        field=(F.field()[:,n]).tolist()
        if range[0]==0 and range[1]==0:
            field_min=min(field)
            field_max=max(field)
            if mode=='sym':
                m=max(abs(field_min),abs(field_max))
                field_min=-m
                field_max=m
        else:
            field_min=range[0]
            field_max=range[1]
        llm=list(zip(lons,lats,field))
        cmap=matplotlib.cm.get_cmap('bwr')
        for p in llm:
            x1,y1 = self._map(p[0]-0.5,p[1]-0.5)
            x2,y2 = self._map(p[0]+0.5,p[1]-0.5)
            x3,y3 = self._map(p[0]+0.5,p[1]+0.5)
            x4,y4 = self._map(p[0]-0.5,p[1]+0.5)
            x=(p[2]-field_min)/(field_max-field_min)
            color=cmap(x)
            poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor=color)
            plt.gca().add_patch(poly)

        return(min(field),max(field))

    def plotgrid(self,G):
        points=G.points()
        lons=points[:,0].tolist()
        lats=points[:,1].tolist()
        x,y=self._map(lons,lats)
        self._map.plot(x,y,'bo', markersize=1)

    def finish(self):
        plt.title(self._title)
        if (self._out=='screen'):
            plt.show()
        else:
            plt.savefig(self._out)
        plt.close()





