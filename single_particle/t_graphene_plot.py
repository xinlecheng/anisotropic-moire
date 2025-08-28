import sys
import pathlib
PORJ_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(PORJ_DIR.as_posix())
import numpy as np
from numpy import pi
import reciprocal_lattice
import os
import general.plot_function as plot_function

DTYPE_REAL = np.float64
DTYPE_COMPLEX = np.complex128

if __name__ == "__main__":
    fracktocar = np.transpose([[0.5, -np.sqrt(3)/2],[0.5, np.sqrt(3)/2]])
    folder_path = '/home/xcheng/Desktop/phD/moire/anisotropic_moire/data/compare_continuum_tb/continuum_0d15'
    if os.path.exists(folder_path) is False:
        os.mkdir(folder_path)
    band_data_m = np.load(os.path.join(folder_path, 'band_data_m.npy'), allow_pickle=False)
    band_data_p = np.load(os.path.join(folder_path, 'band_data_p.npy'), allow_pickle=False)
    band_minima = np.min([np.min(band) for band in band_data_m])
    kpath = [np.array([-1/3, -2/3]), 20, np.array([0,0]), 20, np.array([1/3, 2/3]), 20,
             np.array([2/3, 1/3]), 20, np.array([0,0]), 20, np.array([-2/3, -1/3])]
    klabel = ["K2", "G", "K2'", "K3'", "G", "K3"]
    kticks = reciprocal_lattice.generate_ktick(fracktocar, kpath, klabel)
    print(band_data_m.shape, band_data_p.shape) # (band_index, kpt_index, item), item: (kpath_length, band_energy)
    plot_function.bs_plot(np.concatenate((band_data_m,band_data_p)), kticks,
                             aspect_ratio=0.5, yrange=(-4, 4), color='black', folder=folder_path, suffix='tot') # two valley combined band
    plot_function.bs_plot(band_data_m, kticks, aspect_ratio=0.5, yrange=(-4, 4), color='red', folder=folder_path, suffix='m')
    plot_function.bs_plot(band_data_p, kticks, aspect_ratio=0.5, yrange=(-4, 4), color='blue', folder=folder_path, suffix='p')