# -*- coding: Latin-1 -*-

### Modules ##################################################################
import numpy as np
import scipy as sp
from scipy import interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import os
##############################################################################

##############################################################################
### Plotting #################################################################
##############################################################################


mpl.rcdefaults()
mpl.rcParams['font.size'] = 24. # change the size of the font in every figure
mpl.rcParams['font.family'] = 'Arial' # font Arial in every figure
mpl.rcParams['axes.labelsize'] = 24.
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['xtick.direction'] = "in"
mpl.rcParams['ytick.direction'] = "in"
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.major.width'] = 0.6
mpl.rcParams['ytick.major.width'] = 0.6
mpl.rcParams['axes.linewidth'] = 0.6 # thickness of the axes lines
mpl.rcParams['pdf.fonttype'] = 3  # Output Type 3 (Type3) or Type 42 (TrueType), TrueType allows
                                    # editing the text in illustrator


####################################################
## Plot Parameters #################################

fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure

fig.subplots_adjust(left = 0.18, right = 0.82, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

# axes.axhline(y=0, ls ="--", c ="k", linewidth=0.6)

#############################################
# fig.text(0.79,0.29, "label", ha = "right")
#############################################

axes.set_xlim(0, 1.5)   # limit for xaxis
# axes.set_ylim(top=0) # leave the ymax auto, but fix ymin
axes.set_xlabel(r"$k_{\rm B}T$ / $J$", labelpad = 8)
axes.set_ylabel(r"$\kappa_{\rm xy}$ / $T$", labelpad = 8)

## Allow to shift the label ticks up or down with set_pad
for tick in axes.xaxis.get_major_ticks():
    tick.set_pad(7)
for tick in axes.yaxis.get_major_ticks():
    tick.set_pad(8)


##############################################################################
### Load and Plot Data #######################################################
##############################################################################

######################################################
data = np.loadtxt("kxy_J_1_D_0p05_B_0p01_mesh_100.dat", dtype = "float", comments = "#")

x = data[:,0]
y = -data[:,1]

color = '#3A68FF'

line = axes.plot(x, y / x, label = r"D / J = 0.05")
plt.setp(line, ls ="-", c = color, lw = 3, marker = "", mfc = color, ms = 7, mec = color, mew= 2)
######################################################

######################################################
data = np.loadtxt("kxy_J_1_D_0p1_B_0p01_mesh_100.dat", dtype = "float", comments = "#")

x = data[:,0]
y = -data[:,1]

color = '#FFB628'

line = axes.plot(x, y / x, label = r"D / J = 0.1")
plt.setp(line, ls ="-", c = color, lw = 3, marker = "", mfc = color, ms = 7, mec = color, mew= 2)
######################################################

######################################################
data = np.loadtxt("kxy_J_1_D_0p2_B_0p01_mesh_100.dat", dtype = "float", comments = "#")

x = data[:,0]
y = -data[:,1]

color = 'r'

line = axes.plot(x, y / x, label = r"D / J = 0.2")
plt.setp(line, ls ="-", c = color, lw = 3, marker = "", mfc = color, ms = 7, mec = color, mew= 2)
######################################################

######################################################
plt.legend(loc = 1, fontsize = 12, frameon = False, numpoints=1, markerscale = 1, handletextpad=0.5)
######################################################


##///Set ticks space and minor ticks space ///#
xtics = 0.3 # space between two ticks
mxtics = xtics / 2.  # space between two minor ticks

majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

axes.xaxis.set_major_locator(MultipleLocator(xtics))
axes.xaxis.set_major_formatter(majorFormatter)
axes.xaxis.set_minor_locator(MultipleLocator(mxtics))
# axes.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

axes.locator_params(axis = 'y', nbins = 6)

######################################################

script_name = os.path.basename(__file__)
figurename = script_name[0:-3] + ".pdf"

plt.show()
fig.savefig(figurename, bbox_inches = "tight")
plt.close()
