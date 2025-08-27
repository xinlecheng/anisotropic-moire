import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from reciprocal_lattice import Bilayer, MoireSystem
import numpy as np
import os

def visualize_bilayer(bilayer: Bilayer):
    '''
    visualize the bilayer reciprocal lattice
    '''
    from matplotlib.patches import Circle
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlabel('k_x')
    ax.set_ylabel('k_y')
    ax.set_title('Bilayer Reciprocal Lattice')
    lattice_car = bilayer.lattice_car()
    def color_map(layerlabel: int) -> str:
        return 'red' if layerlabel == 0 else 'green'
    for i, site in enumerate(bilayer.lattice_car()):
        layerlabel = bilayer.lattice[i].layerlabel
        ax.add_patch(Circle(site, 0.1, color=color_map(layerlabel), alpha=0.5))
    # Draw the reciprocal lattice vectors
    xlim = np.max(np.abs(lattice_car[:,0])) + 0.2
    ylim = np.max(np.abs(lattice_car[:,1])) + 0.2
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    plt.grid()
    plt.show()

def visulaize_moire(moire_system: MoireSystem):
    '''
    visualize the moire system with moire hamiltonian as hopping
    '''
    lattice_car = moire_system.bilayer.lattice_car()
    moire_ham = moire_system.moire_ham(moire_system.bilayer)
    from matplotlib.patches import Circle
    def color_map(layerlabel: int) -> str:
        return 'red' if layerlabel == 0 else 'green'
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlabel('k_x')
    ax.set_ylabel('k_y')
    ax.set_title('Moire System Reciprocal Lattice')
    for i, site in enumerate(lattice_car):
        layerlabel = moire_system.bilayer.lattice[i].layerlabel
        ax.add_patch(Circle(site, 0.1, color=color_map(layerlabel), alpha=0.5))
    # Draw the reciprocal lattice vectors
    for i in range(len(lattice_car)):
        for j in range(i+1, len(lattice_car)):
            if moire_ham[i, j] != 0:
                ax.plot([lattice_car[i, 0], lattice_car[j, 0]], 
                        [lattice_car[i, 1], lattice_car[j, 1]], 
                        color='blue', alpha=0.5)
    # Set limits for the plot
    xlim = np.max(np.abs(lattice_car[:,0])) + 0.2
    ylim = np.max(np.abs(lattice_car[:,1])) + 0.2
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    plt.grid()
    plt.show()

def list_plot(datasets, aspect_ratio=1.0, yrange=None, color=None):
    plt.figure()
    if hasattr(datasets[0][0], '__len__'):
        for data in datasets:
            x, y  = zip(*data)
            plt.plot(x, y, marker=None, color=color)
    else:
        x, y  = zip(*datasets)
        plt.plot(x, y, marker=None, color=color)
    if yrange != None:
        plt.ylim(yrange[0], yrange[1])
    plt.gca().set_aspect(aspect_ratio)
    plt.show()
    plt.close()

def bs_plot(bs, kticks, aspect_ratio=1.0, yrange=None, color=None, folder=None, suffix=''):
    plt.figure(figsize=(12,9))
    if hasattr(bs[0][0], '__len__'):
        for data in bs:
            x, y  = zip(*data)
            plt.plot(x, y, marker=None, color=color)
    else:
        x, y  = zip(*bs)
        plt.plot(x, y, marker=None, color=color)
    if yrange != None:
        plt.ylim(yrange[0], yrange[1])
    plt.gca().set_aspect(aspect_ratio)
    plt.ylabel("Energy(eV)")
    plt.xticks(*kticks)
    if folder == None:
        plt.show()
    else:
        os.makedirs(folder, exist_ok=True)
        filename = f"bs_{suffix}.png"
        save_path = os.path.join(folder, filename)
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()
