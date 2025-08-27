import numpy as np
from numpy import pi
from typing import List, Tuple, Union, Optional
import reciprocal_lattice
from reciprocal_lattice import Bilayer, MoireSystem
import plot_function
import os

DTYPR_REAL = np.float64
DTYPR_COMPLEX = np.complex128

# define the pauli matrices
s0 = np.array([[1,0],[0,1]])
sx = np.array([[0,1],[1,0]])
sy = np.array([[0,-1j],[1j,0]])
sv = np.array([sx, sy])
rot = np.array([[0,1],[-1,0]]) # rotation clockwise by pi/2


def kinetic_ham(bilayer: Bilayer, k_car: np.ndarray) -> np.ndarray:
    '''
    construct the hamiltonian for kinetic energy, acts as a confining potential in reciprocal space
    shape of lattice_car: (N, 2), shape of k_car: (2)
    '''
    theta = 1.0 # twist angle/magical angle
    vF = np.sqrt(3)*theta # effective mass, sqrt(3) comes from the fact that norm(q_j) = 1/np.sqrt(3)
    metric = np.array([[1,0],[0,1]]) # metric tensor, identity for now
    lv_car = bilayer.lv_car() # shape: (num_lv, 2)
    gamma_shift = 0.276*np.matmul(bilayer.fracktocar, np.array([1,1])) # shift of the gamma point
    num_lv = len(lv_car)
    ns_plv = bilayer.num_sites_plv
    blocks = np.einsum('ni,ijk->njk', np.matmul(k_car + gamma_shift - lv_car, metric), sv) # shape: (num_site, 2, 2)
    id_mat = np.eye(num_lv) # shape: (num_site, num_site)
    bd:np.ndarray = np.einsum('mn,njk->mnjk', id_mat, blocks)
    return vF*bd.transpose(0,2,1,3).reshape((ns_plv*num_lv, ns_plv*num_lv)) # shape: (2*num_site, 2*num_site)

def moire_ham(bilayer: Bilayer, w0: Union[float, np.ndarray] = np.array([0.577, 0.577, 0.577]), 
              w1: Union[float, np.ndarray] = np.array([0.577, 0.577, 0.577]) ) -> np.ndarray:
    '''
    construct the moire hamiltonian which is are short range hoppings on reciprocal pattice
    '''
    if isinstance(w0, float):
        w0v = np.array([w0,w0,w0]).astype(DTYPR_COMPLEX)
    elif isinstance(w0, np.ndarray):
        w0v = w0
    if isinstance(w1, float):
        w1v = np.array([w1,w1,w1]).astype(DTYPR_COMPLEX)
    elif isinstance(w1, np.ndarray):
        w1v = w1
    qv = np.array([[0, 0],[-np.sqrt(3)/2, -1.5],[np.sqrt(3)/2, -1.5]])/np.sqrt(3) +\
          np.matmul(bilayer.fracktocar, bilayer.layer_dis[0] - bilayer.layer_dis[1])
    lv_car = bilayer.lv_car()
    num_lv = len(lv_car)
    ns_plv = bilayer.num_sites_plv
    k_diff = lv_car.reshape((lv_car.shape[0], 1, lv_car.shape[1])) -\
             lv_car.reshape((1, lv_car.shape[0], lv_car.shape[1]))
    mask_q0 = np.logical_or(np.isclose(k_diff, qv[0]).all(axis=-1), np.isclose(k_diff, -qv[0]).all(axis=-1))
    mask_q1 = np.logical_or(np.isclose(k_diff, qv[1]).all(axis=-1), np.isclose(k_diff, -qv[1]).all(axis=-1))
    mask_q2 = np.logical_or(np.isclose(k_diff, qv[2]).all(axis=-1), np.isclose(k_diff, -qv[2]).all(axis=-1))
    
    site_mask = np.zeros((len(lv_car), len(lv_car)))
    site_mask[mask_q0] = 1
    block_0:np.ndarray = np.einsum('mn,ij->mnij', site_mask, 
                            w0v[0]*s0 + w1v[0]*np.einsum('i,ijk->jk', np.matmul(rot, qv[0]*np.sqrt(3)), sv))
    site_mask[mask_q0] = 0
    site_mask[mask_q1] = 1
    block_1:np.ndarray = np.einsum('mn,ij->mnij', site_mask, 
                            w0v[1]*s0 + w1v[1]*np.einsum('i,ijk->jk', np.matmul(rot, qv[1]*np.sqrt(3)), sv))
    site_mask[mask_q1] = 0
    site_mask[mask_q2] = 1
    block_2:np.ndarray = np.einsum('mn,ij->mnij', site_mask, 
                            w0v[2]*s0 + w1v[2]*np.einsum('i,ijk->jk', np.matmul(rot, qv[2]*np.sqrt(3)), sv))
    
    moire_ham = block_0.transpose(0,2,1,3).reshape((ns_plv*num_lv, ns_plv*num_lv))+\
                block_1.transpose(0,2,1,3).reshape((ns_plv*num_lv, ns_plv*num_lv))+\
                block_2.transpose(0,2,1,3).reshape((ns_plv*num_lv, ns_plv*num_lv))
    return moire_ham

def moire_ham_phi(bilayer: Bilayer) -> np.ndarray:
    '''
    construct the moire hamiltonian based on the T_atom(q)
    '''
    w1 = 0.577
    w0 = 1.0*w1 # 0.7*w1 from congregation
    qv = np.array([[0, 0],[-np.sqrt(3)/2, -1.5],[np.sqrt(3)/2, -1.5]])/np.sqrt(3) +\
          np.matmul(bilayer.fracktocar, bilayer.layer_dis[0] - bilayer.layer_dis[1])
    w0v = w0*np.exp(2*pi**2/3*(1/3 - np.linalg.norm(qv, axis=-1)**2))
    w1v = w1*np.exp(2*pi**2/3*(1/3 - np.linalg.norm(qv, axis=-1)**2))
    #qv_dot_delta_m = np.matmul(qv, np.array([4*pi/3, 0]))
    lv_car = bilayer.lv_car()
    num_lv = len(lv_car)
    ns_plv = bilayer.num_sites_plv
    k_diff = lv_car.reshape((lv_car.shape[0], 1, lv_car.shape[1])) -\
             lv_car.reshape((1, lv_car.shape[0], lv_car.shape[1]))
    mask_q0 = np.logical_or(np.isclose(k_diff, qv[0]).all(axis=-1), np.isclose(k_diff, -qv[0]).all(axis=-1))
    mask_q1 = np.logical_or(np.isclose(k_diff, qv[1]).all(axis=-1), np.isclose(k_diff, -qv[1]).all(axis=-1))
    mask_q2 = np.logical_or(np.isclose(k_diff, qv[2]).all(axis=-1), np.isclose(k_diff, -qv[2]).all(axis=-1))
    
    site_mask = np.zeros((len(lv_car), len(lv_car)))
    site_mask[mask_q0] = 1
    block_0:np.ndarray = np.einsum('mn,ij->mnij', site_mask, 
                            w0v[0]*s0 + w1v[0]*np.einsum('i,ijk->jk', np.array([1, 0]), sv))
    site_mask[mask_q0] = 0
    site_mask[mask_q1] = 1
    block_1:np.ndarray = np.einsum('mn,ij->mnij', site_mask, 
                            w0v[1]*s0 + w1v[1]*np.einsum('i,ijk->jk', np.array([-0.5, np.sqrt(3)/2]), sv))
    site_mask[mask_q1] = 0
    site_mask[mask_q2] = 1
    block_2:np.ndarray = np.einsum('mn,ij->mnij', site_mask, 
                            w0v[2]*s0 + w1v[2]*np.einsum('i,ijk->jk', np.array([-0.5, -np.sqrt(3)/2]), sv))
    
    moire_ham = block_0.transpose(0,2,1,3).reshape((ns_plv*num_lv, ns_plv*num_lv))+\
                block_1.transpose(0,2,1,3).reshape((ns_plv*num_lv, ns_plv*num_lv))+\
                block_2.transpose(0,2,1,3).reshape((ns_plv*num_lv, ns_plv*num_lv))
    return moire_ham


if __name__ == '__main__':
    phi = 0.05
    saving = False
    fracktocar = np.transpose([[0.5, -np.sqrt(3)/2],[0.5, np.sqrt(3)/2]])
    layer_dis_m = np.array([[1/3,2/3],[2/3,1/3]])*(1-3*phi/2/pi)
    layer_dis_p = np.array([[1/3,2/3],[2/3,1/3]])*(1+3*phi/2/pi)
    tbg_bilayer_m = Bilayer(fracktocar, layer_dis_m, 2, 5)
    tbg_bilayer_p = Bilayer(fracktocar, layer_dis_p, 2, 5)
    tbg_moire_m = MoireSystem(tbg_bilayer_m, kinetic_ham, moire_ham_phi)
    tbg_moire_p = MoireSystem(tbg_bilayer_p, kinetic_ham, moire_ham_phi)
    #print(f"lattice: {tbg_moire.bilayer.lattice_car()}")
    plot_function.visualize_bilayer(tbg_bilayer_m)
    #print(f"matrix: {tbg_moire.hamiltonian(np.array([0,0]))[2:4,2:4]}")
    plot_function.visulaize_moire(tbg_moire_m)
    kpath = [np.array([-1/3, -2/3]), 20, np.array([0,0]), 20, np.array([1/3, 2/3]), 20,
             np.array([2/3, 1/3]), 20, np.array([0,0]), 20, np.array([-2/3, -1/3])]
    band_data_m = reciprocal_lattice.bandstructure(tbg_moire_m, kpath)
    band_data_p = reciprocal_lattice.bandstructure(tbg_moire_p, kpath)
    if saving == False:
        band_minima = np.min([np.min(band) for band in band_data_m])
        klabel = ["K2", "G", "K2'", "K3'", "G", "K3"]
        kticks = reciprocal_lattice.generate_ktick(fracktocar, kpath, klabel)
        print(band_data_m.shape, band_data_p.shape) # (band_index, kpt_index, item), item: (kpath_length, band_energy)
        plot_function.bs_plot(np.concatenate((band_data_m,band_data_p)), kticks,
                                aspect_ratio=0.5, yrange=(-4, 4), color='black') # two valley combined band
        plot_function.bs_plot(band_data_m, kticks, aspect_ratio=0.5, yrange=(-4, 4), color='red')
        plot_function.bs_plot(band_data_p, kticks, aspect_ratio=0.5, yrange=(-4, 4), color='blue')
    else:
        folder_path = '/home/xcheng/Desktop/phD/moire/anisotropic_moire/data/compare_continuum_tb/continuum_0d15'
        if os.path.exists(folder_path) is False:
            os.mkdir(folder_path)
        np.save(os.path.join(folder_path, 'band_data_m.npy'), band_data_m)
        np.save(os.path.join(folder_path, 'band_data_p.npy'), band_data_p)