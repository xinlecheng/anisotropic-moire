# Python script to calculate magneto-phonon Langevin dynamics
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
from scipy import integrate


# plt.clf()
# plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rc('font', **{'family':'serif', 'serif':['Computer Modern Roman'], 'monospace':['Computer Modern Typewriter']})
# params = {'backend': 'pdf', 'text.latex.preamble': r"\usepackage{amsmath, amsfonts, amssymb} \usepackage{dsfont} \usepackage{xcolor}", 'axes.labelsize': 22, 'font.size': 20, 'legend.fontsize': 19,
#           'xtick.labelsize': 20, 'ytick.labelsize': 20, 'axes.titlepad' : 12, 'text.usetex': True, 'axes.unicode_minus': True}                                                                                                                                       
# plt.rcParams.update(params)
# mpl.rcParams['pdf.fonttype'] = 42

fig = plt.figure(figsize=(7, 5))
gs = GridSpec(1, 1)
gs.update(left=0.18, right=0.97, bottom=0.17, top=0.95, hspace=0.25, wspace=0.32)

ax1 = fig.add_subplot(gs[0,0])

# Configure plots
n = 20
rcolors = plt.cm.get_cmap('Reds')(np.linspace(0,1,n))
bcolors = plt.cm.get_cmap('Blues')(np.linspace(0,1,n))
gcolors = plt.cm.get_cmap('Greens')(np.linspace(0,1,n))


# Superconductor parameters
omD  = 1               # Debye frequency, meV
nef  = 4.35*10**(21)     # DOS, 1/(eV*cm**3)
D0   = 1.74              # Delta0, meV
leph = 0.19              # Dimensionless electron-phonon coupling
leph = 1.0/np.log(2.0*omD/D0)

# DOS, 1/(meV*nm**3)
nef  = nef*10**(-3)*10**(-7*3)
#print(nef)

# Coupling parameters
d0   = 0.5               # Superconductor-hBN distance for coupling, nm
x    = 0.5 

mev2ev = 1000
ev2mev = 0.001

# Detuning
nd  = 200
det = np.linspace(-0.3, 0.3, nd)
#det = np.linspace(-100.0, 100.0, nd)




# Numerical results
V0 = leph
D0 = 2.0*omD*np.exp(-1.0/V0)

# Phonon-polariton coupling (meV)
lphp = np.array([1.0, 2.0, 3.0])
nl = len(lphp)

for j in range(nl):

    V1 = np.zeros(nd)
    D1 = np.zeros(nd)
    
    ax1.axvline(0.0, linestyle="--", color="gray")
    #ax1.axhline(0.0, linestyle="--", color="gray")

    # Xinle's expressions

    for i in range(nd):
        
        lphp0 = 0.01*lphp[j]

        om1 = omD + 0.5*det[i] + 0.5*np.sqrt(det[i]**2 + 4*lphp0**2)
        om2 = omD + 0.5*det[i] - 0.5*np.sqrt(det[i]**2 + 4*lphp0**2)

        #print(om1, om2)

        V1[i] = omD/om2*0.5*(np.sqrt(det[i]**2 + 4*lphp0**2) + det[i])/np.sqrt(det[i]**2 + 4*lphp0**2) \
                 + omD/om1*0.5*(np.sqrt(det[i]**2 + 4*lphp0**2) - det[i])/np.sqrt(det[i]**2 + 4*lphp0**2)
    
        D1[i] = 2.0*omD*np.exp(-1.0/V1[i])

    #if (j == 0): ax1.plot(det/omD, (V1-V0), color=gcolors[8], linewidth=2.5, label=r"$\lambda = %s$ meV" % lphp[j])
    #if (j == 1): ax1.plot(det/omD, (V1-V0), color=bcolors[8], linewidth=2.5, label=r"$\lambda = %s$ meV" % lphp[j])
    #if (j == 2): ax1.plot(det/omD, (V1-V0), color=rcolors[8], linewidth=2.5, label=r"$\lambda = %s$ meV" % lphp[j])
    if (j == 0): ax1.plot(det/omD, (V1-V0)/V0, color=gcolors[8], linewidth=2.5, label=r"$\lambda = %s$ meV" % lphp[j])
    if (j == 1): ax1.plot(det/omD, (V1-V0)/V0, color=bcolors[8], linewidth=2.5, label=r"$\lambda = %s$ meV" % lphp[j])
    if (j == 2): ax1.plot(det/omD, (V1-V0)/V0, color=rcolors[8], linewidth=2.5, label=r"$\lambda = %s$ meV" % lphp[j])

ax1.set_xlabel(r'$\delta/\omega_D$')
ax1.set_ylabel(r'$(\Delta-\Delta_0)/\Delta_0$', labelpad=12)

#ax1.set_yscale('log')
ax1.legend()
plt.savefig('tem/plot_gap_detuning.png', bbox_inches='tight', dpi=300)
plt.show()
