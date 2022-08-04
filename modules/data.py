#!/usr/bin/env python

# std libs
import re
import os.path
import xarray as xr
import numpy as np
import copy
import modules.basic as basic
import modules.geo as geo

###############################################################################
# classes
###############################################################################
NAMES=['NorESM1-ME', 'GFDL-ESM2M', 'GFDL-ESM2G', 'CESM1-BGC', 'MPI-ESM-LR', 'IPSL-CM5A-MR', 'IPSL-CM5A-LR', 'HadGEM2-CC', 'HadGEM2-ES', 'MIROC-ESM-CHEM', 'MIROC-ESM']

###############################################################################
# classes
###############################################################################
class grid:
    def __init__(self, data):
        if not data: raise Exception("Grid: Initialization error, provide data")
        if data[0]=='file': self.__init__grid__file__(data[1])
        elif data[0]=='cast': self.__init__grid__cast__(data[1])
        elif data[0]=='mask': self.__init__grid__mask__(data[1])
        else: raise Exception("Grid: Initialization error, unknow mode")

    def __init__grid__file__(self,grid_file):
        if not os.path.isfile(grid_file):
            raise Exception("Grid: Initialization error, file '%s' not found" % grid_file)
        self._grid_file=grid_file
        g=xr.open_dataset(self._grid_file)

        self._points=np.hstack((g['lon'].values.reshape(-1,1),g['lat'].values.reshape(-1,1)))
        self._points.flags.writeable=False

        self._area=g['area'].values.reshape(-1,1)
        self._area.flags.writeable=False

    def __init__grid__cast__(self, obj):
        for attr in self.attributes():
            self.__dict__[attr]=obj.__dict__[attr]

    def __init__grid__mask__(self, M):
        self.__init__grid__cast__(M)
        self._grid_file=self._grid_file+" (masked)"
        self._points=self._points[M.mask(),:]
        self._area=self._area[M.mask(),:]

    def attributes(self):
        return(['_grid_file','_points','_area'])

    def n(self):
        return(self._points.shape[0])

    def points(self):
        return(self._points)

    def area(self):
        return(self._area)

    def plot_grid(self, grid=[], out='screen', title=''):
        m=geo.map(grid=[], out=out, title='Mask')
        m.plotgrid(self)
        m.finish()

    def plot(self, grid=[], out='screen', title=''):
        self.plot_grid(grid=grid, out=out, title=title)

    def __str__(self):
        s="--- grid ---\n"
        s=s+"file: %s\n" % self._grid_file
        s=s+"nodes: %d\n" % self.n()
        return(s)

    def __hash__(self):
        h=hash("grid")+basic.numpy_array_hash(self._points)+basic.numpy_array_hash(self._area)
        return(h)

    def __eq__(self, other):
        if self.__hash__()==other.__hash__(): return(True)
        return(False)


class field(grid):
    def __init__(self, data):
        if not data: raise Exception("Field: Initialization error, provide data")
        if data[0]=='file': self.__init__field__file__(data[1],data[2])
        elif data[0]=='mask': self.__init__field__mask__(data[1],data[2])
        elif data[0]=='mask_stat': self.__init__field__mask_stat__(data[1])
        elif data[0]=='all': self.__init__field__all__()
        else: raise Exception("Field: Initialization error, unknow mode")

    def __init__field__file__(self, grid_file, field_file):
        super(field,self).__init__(['file',grid_file])
        if not os.path.isfile(field_file):
            raise Exception("Field: Initialization error, file '%s' not found" % file)
        self._field_files=[field_file]
        f=xr.open_dataset(self._field_files[0])

        self._field=f['delta_spco2'].values.reshape(-1,1)
        self._field.flags.writeable=False

        if self._field.shape[0]!=grid.n(self):
            raise Exception("Field: Initialization error, '%s' is not compatible with the grid provided" % self._field_files[0])

        for p in NAMES:
            n=re.findall(p+'_',self._field_files[0])
            if n: break
        if not p:
            raise Exception("Field: Initialization error, '%s' cannot associated with a known name" % self._field_files[0])
        self._names=[p]

    def __init__field__mask__(self, F, M):
        super(field,self).__init__(['mask',M])
        self._field_files=[f+' (masked)' for f in F._field_files]
        self._field=F._field[M.mask(),:]
        self._field.flags.writeable=False
        self._names=F._names.copy()

    def __init__field__mask_stat__(self, M):
        M=list(M)
        super(field,self).__init__(['cast',M[0]])
        if len(M)<2:
            raise Exception("Field: Initialization error, provide at least two masks")
        for i in range(1,len(M)):
            if grid(['cast',M[0]])!=grid(['cast',M[i]]):
                raise Exception("Field: Initialization error, grids of set of masks have to be equal")
        self._field_files=['from {M}','from {M}']
        A=M[0].mask().reshape(-1,1)
        for i in range(1,len(M)):
            A=np.hstack((A,M[i].mask().reshape(-1,1)))
        self._field=np.hstack((np.average(A,axis=1).reshape(-1,1),np.std(A,axis=1,ddof=1).reshape(-1,1)))
        self._field.flags.writeable=False
        self._names=['avg','std']

    def __init__field__all__(self):
        F0=field(['file','data/one_degree_grid_NA.nc','data/CESM1-BGC_spco2_NA_1990s_summer_anomaly_onedegree.nc'])
        F0=F0+field(['file','data/one_degree_grid_NA.nc','data/GFDL-ESM2G_spco2_NA_1990s_summer_anomaly_onedegree.nc'])
        F0=F0+field(['file','data/one_degree_grid_NA.nc','data/GFDL-ESM2M_spco2_NA_1990s_summer_anomaly_onedegree.nc'])
        F0=F0+field(['file','data/one_degree_grid_NA.nc','data/HadGEM2-CC_spco2_NA_1990s_summer_anomaly_onedegree.nc'])
        F0=F0+field(['file','data/one_degree_grid_NA.nc','data/HadGEM2-ES_spco2_NA_1990s_summer_anomaly_onedegree.nc'])
        F0=F0+field(['file','data/one_degree_grid_NA.nc','data/IPSL-CM5A-LR_spco2_NA_1990s_summer_anomaly_onedegree.nc'])
        F0=F0+field(['file','data/one_degree_grid_NA.nc','data/IPSL-CM5A-MR_spco2_NA_1990s_summer_anomaly_onedegree.nc'])
        F0=F0+field(['file','data/one_degree_grid_NA.nc','data/MIROC-ESM-CHEM_spco2_NA_1990s_summer_anomaly_onedegree.nc'])
        F0=F0+field(['file','data/one_degree_grid_NA.nc','data/MIROC-ESM_spco2_NA_1990s_summer_anomaly_onedegree.nc'])
        F0=F0+field(['file','data/one_degree_grid_NA.nc','data/MPI-ESM-LR_spco2_NA_1990s_summer_anomaly_onedegree.nc'])
        F0=F0+field(['file','data/one_degree_grid_NA.nc','data/NorESM1-ME_spco2_NA_1990s_summer_anomaly_onedegree.nc'])

        # filter only the non-zeros (zero is encoding of invalid data)
        M0=mask(['field',F0,lambda x: x!=0,'and'])
        F1=field(['mask',F0,M0])
        M1=mask(['field',F1,8.47,'stddev'])
        F2=field(['mask',F1,M1])
        for key in F2.__dict__:
            self.__dict__[key]=F2.__dict__[key]

    def field(self):
        return(self._field)

    def m(self):
        return(self._field.shape[1])

    def names(self):
        return(self._names)

    def copy(self):
        return(copy.deepcopy(self))

    def savetxt(self,file,meta=''):
        p=self.points()
        f=self._field[:,0].reshape(-1,1)
        if meta:
            np.savetxt(file,np.hstack((p,f)),comments='# ',header=meta)
        else:
            np.savetxt(file,np.hstack((p,f)))

    def plot_field(self, comp, range=[0,0], mode='sym', grid=[], out='screen', title=''):
        m=geo.map(grid=[], out=out, title=title)
        m.plotfield(self, comp, range=range, mode=mode)
        m.finish()

    def plot(self, comp, range=[0,0], mode='sym', grid=[], out='screen', title=''):
        self.plot_field(comp, range=range, mode=mode, grid=grid, out=out, title=title)

    def __add__(self,other):
        if super(field,self).__hash__()!=super(field,other).__hash__():
            raise Exception("Field: cannot add field over different grids")
        sum=self.copy()
        sum._field=np.hstack((sum._field,copy.deepcopy(other._field)))
        sum._field_files=sum._field_files+other._field_files
        sum._names=sum._names+other._names
        return(sum)

    def __str__(self):
        s="--- field ---\n"
        s=s+"nodes: %d\n" % self.n()
        s=s+"components: %d\n" % self.m()
        names_len_max=max([len(name) for name in self._names])+1
        for i in range(self.m()):
            name=self._names[i]+':'
            format_string='%-'+str(names_len_max)+"s %s, "
            s=s+format_string % (name,self._field_files[i])
            s=s+"range: [%f, %f]\n" % (np.min(self._field[:,i]),np.max(self._field[:,i]))
        s=s+"\n"
        s=s+grid.__str__(self)
        return(s)

    def __hash__(self):
        h=hash("field")+basic.numpy_array_hash(self._field)+super(field,self).__hash__()
        return(h)

    def __eq__(self, other):
        if self.__hash__()==other.__hash__(): return(True)
        return(False)


class mask(grid):
    def __init__(self, data):
        if not data: raise Exception("Mask: Initialization error, provide data")
        if data[0]=='field': self.__init__mask__field__(data[1],data[2],data[3])
        elif data[0]=='mask': self.__init__mask__mask__(data[1])
        elif data[0]=='m': self.__init__mask__m__(data[1],data[2])
        elif data[0]=='sig': self.__init__mask__sig__(data[1],data[2])
        else: raise Exception("Field: Initialization error, unknow mode")

    def __init__mask__field__(self, F, fct, mode):
        super(mask,self).__init__(['cast',F])
        X=F.field()
        if len(X.shape)!=2:
            raise Exception("Mask: can create mask only from 2d-arrays")
        if X.shape[0]!=self.n():
            raise Exception("Mask: grid is inconsistent with F")
        if mode=='and':
            self._A=np.full((X.shape[0]), True, dtype=bool)
            for n in range(X.shape[0]):
                for m in range(X.shape[1]):
                    if fct(X[n,m])==False: self._A[n]=False
        if mode=='or':
            self._A=np.full((X.shape[0]), False, dtype=bool)
            for n in range(X.shape[0]):
                for m in range(X.shape[1]):
                    if fct(X[n,m])==True: self._A[n]=True
        if mode=='first':
            self._A=np.full((X.shape[0]), False, dtype=bool)
            for n in range(X.shape[0]):
                if fct(X[n,0])==True: self._A[n]=True
        if mode=='stddev':
            self._A=np.std(X,axis=1,ddof=1)>fct

    def __init__mask__mask__(self, M):
        super(mask,self).__init__(['mask',M])
        self._A=M._A[M.mask()]

    def __init__mask__m__(self, M, m):
        super(mask,self).__init__(['cast',M])
        self._A=m

    def __init__mask__sig__(self, G, s):
        super(mask,self).__init__(['cast',G])
        self._A=np.full(G.n(), False, dtype=bool)
        for i in range(len(s)):
            if s[i]=='1': self._A[i]=True
            else: self._A[i]=False

    def active(self):
        return(np.sum(self._A))

    def inactive(self):
        return(self.n()-self.active())

    def mask(self):
        return(self._A)

    def set_mask(self,m):
        self._A=m

    def sig(self):
        return(''.join([str(int(x)) for x in self._A]))

    def savetxt(self,file,meta=''):
        p=self.points()
        m=self.mask()
        if meta:
            np.savetxt(file,p[m],comments='# ',header=meta)
        else:
            np.savetxt(file,p[m])

    def plot_mask(self, grid=[], out='screen', title=''):
        m=geo.map(grid=[], out=out, title=title)
        m.plotmask(self)
        m.finish()

    def plot(self, grid=[], out='screen', title=''):
        self.plot_mask(grid=grid, out=out, title=title)

    def __str__(self):
        s="--- mask ---\n"
        s=s+"active: %d\n" % self.active()
        s=s+"inactive: %d\n\n" % self.inactive()
        s=s+super(mask,self).__str__()
        return(s)

    def __hash__(self):
        h=hash("mask")+basic.numpy_array_hash(self._A)+super(mask,self).__hash__()
        return(h)

    def __eq__(self, other):
        if self.__hash__()==other.__hash__(): return(True)
        return(False)










