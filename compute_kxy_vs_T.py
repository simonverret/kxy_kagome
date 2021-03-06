# -*- coding: Latin-1 -*-

## Modules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
import numpy as np
from numpy import pi, sqrt
np.set_printoptions(6,suppress=True,sign="+",floatmode="fixed")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from package_kxy.bands_functions import *
import time
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

## Parameters :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
J = 1
B = 0.01 * J
D = 0.01 * J

resolution_k = 100
T_array = np.arange(0.02, 1.3, 0.01)[::-1]

# Initial parameters for root algorithm :::::::::::::::::::::::::::::::::::::::#
chi_up_ini = -0.38
chi_dn_ini = -0.37

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
# Contains the hexago volume in a rectangle of side pi x (2*pi/sqrt(3))
kx = np.linspace(-pi / 3, 2 * pi / 3, resolution_k)
ky = np.linspace(-pi / sqrt(3), pi / sqrt(3), resolution_k)
kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
## Initialize guessed values for root function ##
ts_up_ini = compute_ts(chi_up_ini, chi_dn_ini, J, D, 1)
ts_dn_ini = compute_ts(chi_up_ini, chi_dn_ini, J, D, -1)

la_min_up = diag_func(kxx, kyy, la = 0, s = 1, B = B, ts = ts_up_ini)[-1]
la_min_dn = diag_func(kxx, kyy, la = 0, s = -1, B = B, ts = ts_dn_ini)[-1]

la_min = np.max([la_min_up, la_min_dn]) # prevent from having negative eigen values in the root algorithm

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
## Compute Kxy ##
kxy_array = np.empty(T_array.shape)
la_array = np.empty(T_array.shape)
chi_up_array = np.empty(T_array.shape)
chi_dn_array = np.empty(T_array.shape)
B_array = np.ones(T_array.shape) * B
J_array = np.ones(T_array.shape) * J
D_array = np.ones(T_array.shape) * D
res_array = np.ones(T_array.shape) * resolution_k

for i, t in enumerate(T_array):

    start_time_kxy = time.time()

    kxy, la, chi_up, chi_dn = kxy_algorithm(kxx, kyy, B, D, J, t, la_min, chi_up_ini, chi_dn_ini)[:-2]
    (la_min, chi_up_ini, chi_dn_ini) = (la, chi_up, chi_dn) # change the initial values for next T

    kxy_array[i] = kxy
    la_array[i] = la
    chi_up_array[i] = chi_up
    chi_dn_array[i] = chi_dn

    print("One value kxy time : %.6s seconds" % (time.time() - start_time_kxy))


## Save Data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
Data = np.vstack((T_array/J, B_array, kxy_array, la_array, chi_up_array, chi_dn_array, J_array, D_array, res_array))
Data = Data.transpose()
folder = "../data_sim/"
file_name =  "TS-kxy" + "_B_" + str(B) + "_J_" + str(J) + "_D_" + str(D) + "_res_" + str(resolution_k) + ".dat"
np.savetxt(folder + file_name, Data, fmt='%.7e', header = "T/J\tB\tkxy\tla\tchi_up\tchi_dn\tJ\tD\tresolution_k", comments = "#")

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
# Figures PRINT ///////////////////////////////////////////////////////////////#
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

#///// RC Parameters //////#
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


#>>>> Kxy vs T >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
fig.subplots_adjust(left = 0.18, right = 0.82, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

axes.axhline(y = 0, ls ="--", c ="k", linewidth = 0.6)

#///// Allow to shift the label ticks up or down with set_pad /////#
for tick in axes.xaxis.get_major_ticks():
    tick.set_pad(7)
for tick in axes.yaxis.get_major_ticks():
    tick.set_pad(8)

#///// Labels //////#
fig.text(0.84, 0.86, r"$B$ = " + str(B))
fig.text(0.84, 0.79, r"$J$  = " + str(J))
fig.text(0.84, 0.72, r"$D$ = " + str(D))

color = '#29FB87'

line = axes.plot(T_array / J, kxy_array / T_array)
plt.setp(line, ls ="-", c = color, lw = 3, marker = "", mfc = 'k', ms = 8, mec = "#7E2320", mew= 2)  # set properties

axes.set_xlim(0, None)   # limit for xaxis
axes.set_ylim(None, 0) # leave the ymax auto, but fix ymin
axes.set_xlabel(r"$k_{\rm B}T$ / $J$", labelpad = 8)
axes.set_ylabel(r"$\kappa_{\rm xy}$ / $T$", labelpad = 8)


##///Set ticks space and minor ticks space ///#
xtics = 0.3 # space between two ticks
mxtics = xtics / 2.  # space between two minor ticks

majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

axes.xaxis.set_major_locator(MultipleLocator(xtics))
axes.xaxis.set_major_formatter(majorFormatter)
axes.xaxis.set_minor_locator(MultipleLocator(mxtics))
axes.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

axes.locator_params(axis = 'y', nbins = 6)


plt.show()

folder = "../figures_sim/"
figure_name =  "TS-kxy" + "_B_" + str(B) + "_J_" + str(J) + "_D_" + str(D) + "_res_" + str(resolution_k) + ".pdf"
fig.savefig(folder + figure_name, bbox_inches = "tight")

#//////////////////////////////////////////////////////////////////////////////#