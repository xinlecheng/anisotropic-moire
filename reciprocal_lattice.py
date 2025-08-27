import numpy as np
from typing import Dict, List, Tuple, Union
from numpy import pi

DTYPR_REAL = np.float64
DTYPR_COMPLEX = np.complex128

class SiteIndex:
    """
    label an arbitrary site on a reciprocal lattice vector on a given layer,
    layerlabel convention for bilayer: 0 for top, 1 for bottom, top is twisted.
    """
    def __init__(self, layerlabel:int, sitelabel:int, bravis:Union[tuple, np.ndarray]) -> None:
        if isinstance(bravis, tuple):
            self._val = (layerlabel, sitelabel, bravis)
        elif isinstance(bravis, np.ndarray):
            self._val = (layerlabel, sitelabel, tuple(bravis))
        else:
            raise TypeError("wrong type for bravis!")

    @property
    def layerlabel(self) -> int:
        return self._val[0]
    @property
    def sitelabel(self) -> int:
        return self._val[1]
    @property
    def bravis(self) -> np.ndarray:
        return np.array(self._val[2]) #return the bravis vectors as array
    
    def __eq__(self, other):
        if isinstance(other, SiteIndex):
            return self._val == other._val
        else:
            return False
    def __hash__(self) -> int:
        return hash(self._val)
    
    @classmethod
    def from_tuple(cls, val: Tuple[int, int, Tuple[int,...]]) -> "SiteIndex":
        return SiteIndex(val[0], val[1], val[2])

def monolayer_lattice(layerlabel:int, fracktocar:np.ndarray, num_sites_plv:int,
                       cutoff:float, center=np.array([0,0])) -> List[SiteIndex]:
    '''
    construct the 2d reciprocal lattice for monolayer, with a certain cutoff.
    num_sites_pl is the number of sites per lattice vector;
    center: center of the monolayer lattice in fractional coordinates.
    the output ordering is n1 -> n2 -> site
    '''
    k_metric = np.matmul(np.transpose(fracktocar), fracktocar)
    metric_eigs = np.linalg.eigvalsh(k_metric)
    lattice_cutoff = int(np.ceil(cutoff/np.min(metric_eigs)))
    return [SiteIndex(layerlabel, i, (n1, n2)) for n1 in range(-lattice_cutoff, lattice_cutoff+1)
             for n2 in range(-lattice_cutoff, lattice_cutoff+1) for i in range(num_sites_plv)
            if np.linalg.norm(np.matmul(fracktocar, np.array([n1, n2]) - center)) < cutoff]

class Bilayer:
    def __init__(self, fracktocar:np.ndarray, k_dis:np.ndarray, num_sites_plv:int, cutoff:float):
        '''
        fracktocar: coodinate transformation from fractional to cartesian
        k_dis: kspace displacement of top and bottom layers, in fractional coordinates
        num_sites_plv: number os sites per k-lattice vector
        cutoff: cutoff for k-lattice in cartesian measure
        '''
        self.fracktocar = fracktocar
        self.num_sites_plv = num_sites_plv
        self.layer_dis = k_dis #fractional displacement of the top and bottom layers in k space
        self.lv = monolayer_lattice(0, fracktocar, 1, cutoff, center= -k_dis[0]) +\
              monolayer_lattice(1, fracktocar, 1, cutoff, center= -k_dis[1]) # list of lattice vectors, site on each lv is suppressed
        self.top_lattice = monolayer_lattice(0, fracktocar, num_sites_plv, cutoff, center= -k_dis[0])
        self.bot_lattice = monolayer_lattice(1, fracktocar, num_sites_plv, cutoff, center= -k_dis[1])
        self.lattice = self.top_lattice + self.bot_lattice
    def lattice_car(self) -> np.ndarray:
        '''
        return the cartesian coordinates of all lattice sites, with k_dis added;
        output shape: (2*N*num_sites_plv, 2), N is the number of k-lattice per layer.
        '''
        return np.array([np.matmul(self.fracktocar, stind.bravis + self.layer_dis[stind.layerlabel]) for stind in self.lattice])
    def lv_car(self) -> np.ndarray:
        '''
        return the cartesian coordinates of all lattice vectors, with k_dis added,
        sites on each lattice vector is suppressed
        '''
        return np.array([np.matmul(self.fracktocar, stind.bravis + self.layer_dis[stind.layerlabel]) for stind in self.lv]) 

class MoireSystem:
    def __init__(self, bilayer: Bilayer, kinetic_ham, moire_ham):
        self.bilayer = bilayer
        self.kinetic_ham = kinetic_ham
        self.moire_ham = moire_ham
        self.hamiltonian = lambda k_car: self.kinetic_ham(bilayer, k_car) + self.moire_ham(bilayer)
    
def bandstructure(moire:MoireSystem, kpath: list):
    '''
    output the bandstructure of sps along kpath
    '''
    ktocar = moire.bilayer.fracktocar
    kpoints = [np.array(kpath[i-2]) + (np.array(kpath[i])- np.array(kpath[i-2]))*n/kpath[i-1] 
               for i in range(2, len(kpath), 2) for n in range(kpath[i-1])]
    l = 0
    tem = []
    for i in range(1, len(kpoints)):
        dk = np.matmul(ktocar, kpoints[i]-kpoints[i-1])
        l += np.linalg.norm(dk)
        tem.append((l, kpoints[i]))
    kpoints = [(0.0, kpoints[0])] + tem
    tem = []
    for i in range(len(kpoints)):
        v = np.linalg.eigvalsh(moire.hamiltonian(np.matmul(ktocar, kpoints[i][1])))
        tem.append([(kpoints[i][0], v[j]) for j in range(len(v))])
    return np.transpose(tem,(1,0,2))  

def generate_ktick(fracktocar:np.ndarray, kpath: list, klabel: list):
    '''
    generate the ticks used in the bandstructure plot
    '''  
    ktocar = fracktocar
    kpoints = [kpath[i] for i in range(0, len(kpath), 2)] #all kink points
    l = 0
    tem = []
    for i in range(1, len(kpoints)):
        dk = np.matmul(ktocar, kpoints[i]-kpoints[i-1])
        l += np.linalg.norm(dk)
        tem.append((l, kpoints[i]))
    kpoints = [(0.0, kpoints[0])] + tem
    kpt_l = [kpt[0] for kpt in kpoints]
    return [kpt_l, klabel]


