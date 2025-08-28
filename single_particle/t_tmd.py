import sys
import pathlib
PORJ_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(PORJ_DIR.as_posix())
import numpy as np
from numpy import pi
from typing import List, Tuple, Union
import reciprocal_lattice
from reciprocal_lattice import Bilayer, MoireSystem
import general.plot_function as plot_function

DTYPR_REAL = np.float64
DTYPR_COMPLEX = np.complex128

def kinetic_ham(bilayer: Bilayer, k_car: np.ndarray) -> np.ndarray:
    '''
    construct the hamiltonian for kinetic energy, acts as a confining potential in reciprocal space
    '''
    theta = 1.65 # twist angle in degree
    m_eff = 0.0592/theta**2 # effective mass m_eff = meV*m/(pi/180*4pi/sqrt(3)/a)**2/hbar**2, meff ~ 0.1*m/me
    lv_car = bilayer.lv_car()
    return 1/2/m_eff*np.diag(np.linalg.norm(k_car-lv_car, axis=-1)**2).astype(DTYPR_COMPLEX)

def moire_ham(bilayer: Bilayer, w=-18, v=-9, phi=120*pi/180) -> np.ndarray:
    '''
    construct the moire hamiltonian which is the short range hopping on reciprocal pattice
    '''
    lattice_car = bilayer.lattice_car()
    k_diff = lattice_car.reshape((lattice_car.shape[0], 1, lattice_car.shape[1])) -\
             lattice_car.reshape((1, lattice_car.shape[0], lattice_car.shape[1]))
    k_diff_norm = np.linalg.norm(k_diff, axis=-1)
    g_p = [np.array([1,0]), np.array([-0.5, np.sqrt(3)/2]), np.array([-0.5, -np.sqrt(3)/2])]
    g_m = [np.array([-1,0]), np.array([0.5, -np.sqrt(3)/2]), np.array([0.5, np.sqrt(3)/2])]
    mask_nearest = np.logical_and(k_diff_norm > 0.1, k_diff_norm < 1/np.sqrt(3) + 0.1) # assuming that the neareset neighbor in kspace has distance 1/sqrt(3)
    layer_vec = np.array([site.layerlabel for site in bilayer.lattice])
    layer_mask_p = layer_vec == 0
    layer_mask_m = layer_vec == 1
    layer_mask_pp = np.logical_and(layer_mask_p.reshape((-1, 1)), layer_mask_p.reshape((1, -1)))
    layer_mask_mm = np.logical_and(layer_mask_m.reshape((-1, 1)), layer_mask_m.reshape((1, -1)))
    mask_gp = np.logical_or(np.isclose(k_diff, g_p[0]).all(axis=-1),
                                   np.logical_or(np.isclose(k_diff, g_p[1]).all(axis=-1),
                                                 np.isclose(k_diff, g_p[2]).all(axis=-1)))
    mask_gm = np.logical_or(np.isclose(k_diff, g_m[0]).all(axis=-1),
                                   np.logical_or(np.isclose(k_diff, g_m[1]).all(axis=-1),
                                                 np.isclose(k_diff, g_m[2]).all(axis=-1)))
    moire_ham = np.zeros((len(lattice_car), len(lattice_car))).astype(DTYPR_COMPLEX)
    moire_ham[mask_nearest] = w
    moire_ham[np.logical_or(np.logical_and(layer_mask_pp, mask_gp),
                            np.logical_and(layer_mask_mm, mask_gm))] = v*np.exp(1j*phi)
    moire_ham[np.logical_or(np.logical_and(layer_mask_mm, mask_gp),
                            np.logical_and(layer_mask_pp, mask_gm))] = v*np.exp(-1j*phi)
    return moire_ham

if __name__ == '__main__':
    
    fracktocar = np.transpose([[0.5, -np.sqrt(3)/2],[0.5, np.sqrt(3)/2]])
    layer_dis = np.array([[1/3,2/3],[2/3,1/3]])
    tmd_bilayer = Bilayer(fracktocar, layer_dis, 1, 5)
    tmd_moire = MoireSystem(tmd_bilayer, kinetic_ham, moire_ham)
    plot_function.visualize_bilayer(tmd_bilayer)
    plot_function.visulaize_moire(tmd_moire)
    kpath = [np.array([0, 0]), 40, np.array([1/3, 2/3]), 20, np.array([1/2, 1/2]), 40, np.array([0, 0])]
    band_data = reciprocal_lattice.bandstructure(tmd_moire, kpath)
    band_minima = np.min([np.min(band) for band in band_data])
    plot_function.list_plot(band_data, aspect_ratio=0.05, yrange=(band_minima-1, band_minima+80))