# -*- coding: Latin-1 -*-

## Modules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
import numpy as np
from numpy import cos, sin, pi, sqrt, exp
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
from functools import partial
from scipy import optimize
from numpy.linalg import multi_dot

import package_kxy.bands_functions as bands_func
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

## Universal Constant :::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

# hbar = 1.05457173e-34 # J.s
# e = 1.60217657e-19 # coulombs
# kB = 1.380648e-23 # J / K

## Variables ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

B = 0.01 # magnetic field in unit of energy g * muB * B

T = 1
kB = 1

J = 1
D = 0.2 * J

la_ini = 0
chi_up_ini = -1
chi_dn_ini = -1

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

# Contains the hexago volume in a rectangle of side pi x (2*pi/sqrt(3))
kx = np.linspace(-pi / 3, 2 * pi / 3, 10)
ky = np.linspace(-pi / sqrt(3), pi / sqrt(3), 10)

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

def residual_chi(chi, la, kx, ky, B, T):

    chi_up = chi[0]
    chi_dn = chi[1]

    # Compute ts_up & ts_dn
    ts_up = bands_func.compute_ts(chi_up, chi_dn, J, D, 1)
    ts_dn = bands_func.compute_ts(chi_up, chi_dn, J, D, -1)

    # Compute eigenvalues
    Enks_up, Enks_ndiag_up = bands_func.diag_func(kx, ky, la, s = 1, B = B, ts = ts_up)[0:2]
    Enks_dn, Enks_ndiag_dn = bands_func.diag_func(kx, ky, la, s = -1, B = B, ts = ts_dn)[0:2]

    # Compute residual_Chi
    (chi_up_new, chi_dn_new) = bands_func.compute_chi(Enks_up, Enks_dn, Enks_ndiag_up, Enks_ndiag_dn, ts_up, ts_dn, T)
    residual_chi_up = chi_up - chi_up_new
    residual_chi_dn = chi_dn - chi_dn_new

    return (residual_chi_up, residual_chi_dn)

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


##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
## Compute S for different values of lambda with optimized chi_up and chi_dn ###
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#


la_array = np.arange(2.5, 4, 0.2)
S_array = np.zeros(len(la_array))

for i, laa in enumerate(la_array):

    p_residual_chi = partial(residual_chi, la = laa, kx = kx, ky = ky, B = B, T = T)

    # In order to avoid the trivial value for (chi_up, chi_dn) = (0,0), we look for
    # chi roots different from the trivial ones by trying different chi_ini values
    # starting from chi_ini ~ 0 to higher values, as the non trivial roots are the second
    # roots to find before chi_function becomes discontinous:

    chi_steps = np.linspace(0.01, 2, 10)
    for chi_ini in chi_steps:

        sol_object = optimize.root(p_residual_chi, np.array([-chi_ini, -chi_ini]))
        chi_roots = sol_object.x

        if np.any(np.abs(chi_roots) < 1e-2) :
            continue
        else:
            break

    print("for lambda = " + "{0:.2e}".format(laa) + " roots(chi_up, chi_dn ) = " + str(chi_roots))

    ts_up = bands_func.compute_ts(chi_roots[0], chi_roots[1], J, D, 1)
    ts_dn = bands_func.compute_ts(chi_roots[0], chi_roots[1], J, D, -1)

    Enks_up, Enks_ndiag_up, Vnks_up, la_min_up = bands_func.diag_func(kx, ky, laa, s = 1, B = B, ts = ts_up)
    Enks_dn, Enks_ndiag_dn, Vnks_dn, la_min_dn = bands_func.diag_func(kx, ky, laa, s = -1, B = B, ts = ts_dn)

    S_array[i] = bands_func.compute_S(Enks_up, Enks_dn, T)


#///// Plot /////#

fig , axes = plt.subplots(1,1, figsize=(9.2, 5.6)) # figsize is w x h in inch of figure
fig.subplots_adjust(left = 0.17, right = 0.81, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

for tick in axes.xaxis.get_major_ticks():
    tick.set_pad(7)
for tick in axes.yaxis.get_major_ticks():
    tick.set_pad(8)

axes.axhline(y = 0.5, ls = "--", c = "k", linewidth = 0.6)

line = axes.plot(la_array, S_array)
plt.setp(line, ls = "--", c = '#F60000', lw = 3, marker = "o", mfc = 'w', ms = 6.5, mec = '#F60000', mew = 2.5)

axes.locator_params(axis = 'x', nbins = 6)
axes.locator_params(axis = 'y', nbins = 6)
axes.set_xlabel(r"$\lambda$", labelpad = 8)
axes.set_ylabel(r"$S$", labelpad = 8)


##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
## Compute diff_chi_up & dn = 0 in function of chi_up and chi_dn ::::::::::::###
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#


la = 3

span = 0.003

chi_up_array = np.arange(-1, 0, 0.01)
chi_dn_array = np.arange(-1, 0, 0.01)

ts_up_array = np.zeros(np.shape(chi_up_array))
ts_dn_array = np.zeros(np.shape(chi_dn_array))

diff_chi_up = np.zeros((len(chi_up_array), len(chi_dn_array)), dtype = float)
diff_chi_dn = np.zeros((len(chi_dn_array), len(chi_dn_array)), dtype = float)

chi_up_0_x = []
chi_up_0_y = []
chi_dn_0_x = []
chi_dn_0_y = []

n_max_iter = np.size(chi_up_array) * np.size(chi_dn_array)

n = 0
for i, chi_up in enumerate(chi_up_array):
    for j, chi_dn in enumerate(chi_dn_array):

        ts_up = bands_func.compute_ts(chi_up, chi_dn, J, D, 1)
        ts_dn = bands_func.compute_ts(chi_up, chi_dn, J, D, -1)

        Enks_up, Enks_ndiag_up, Vnks_up, la_min_up = bands_func.diag_func(kx, ky, la, s = 1, B = B, ts = ts_up)
        Enks_dn, Enks_ndiag_dn, Vnks_dn, la_min_dn = bands_func.diag_func(kx, ky, la, s = -1, B = B, ts = ts_dn)

        (chi_up_new, chi_dn_new) = bands_func.compute_chi(Enks_up, Enks_dn, Enks_ndiag_up, Enks_ndiag_dn, ts_up, ts_dn, T)

        diff_chi_up[i, j] = chi_up_new - chi_up
        diff_chi_dn[i, j] = chi_dn_new - chi_dn

        if (diff_chi_up[i, j] < span) * (diff_chi_up[i, j] > -span):
            chi_up_0_x.append(chi_up)
            chi_up_0_y.append(chi_dn)

        if (diff_chi_dn[i, j] < span) * (diff_chi_dn[i, j] > -span):
            chi_dn_0_x.append(chi_up)
            chi_dn_0_y.append(chi_dn)

        n += 1
        print("n_iter / n_max = " + str(n) + " / "+ str(n_max_iter))


## Compute the crossing of diff_chi_up & diff_chi_dn

p_residual_chi = partial(residual_chi, la = la, kx = kx, ky = ky, B = B, T = T)

# In order to avoid the trivial value for (chi_up, chi_dn) = (0,0), we look for
# chi roots different from the trivial ones by trying different chi_ini values
# starting from chi_ini ~ 0 to higher values, as the non trivial roots are the second
# roots to find before chi_function becomes discontinous:

chi_steps = np.linspace(0.01, 2, 10)
for chi_ini in chi_steps:

    sol_object = optimize.root(p_residual_chi, np.array([-chi_ini, -chi_ini]))
    chi_roots = sol_object.x

    if np.any(np.abs(chi_roots) < 1e-2) :
        continue
    else:
        break

chi_up_roots = chi_roots[0]
chi_dn_roots = chi_roots[1]



#///// Plot /////#

fig , axes = plt.subplots(1,1, figsize=(9.2, 5.6)) # figsize is w x h in inch of figure
fig.subplots_adjust(left = 0.17, right = 0.81, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

for tick in axes.xaxis.get_major_ticks():
    tick.set_pad(7)
for tick in axes.yaxis.get_major_ticks():
    tick.set_pad(8)

fig.text(0.83, 0.87, r"$\lambda$ = " + str(la), fontsize = 18)
fig.text(0.83, 0.82, "D = " + str(D), fontsize = 18)
fig.text(0.83, 0.77, "J = " + str(J), fontsize = 18)
fig.text(0.83, 0.72, "T = " + str(T), fontsize = 18)
fig.text(0.83, 0.67, "B = " + str(B), fontsize = 18)
fig.text(0.83, 0.62, "size kx = " + str(len(kx)), fontsize = 18)
fig.text(0.83, 0.57, "size ky = " + str(len(ky)), fontsize = 18)

#///// Plot /////#

line = axes.plot(chi_up_0_x, chi_up_0_y)
plt.setp(line, ls = "", c = 'r', lw = 3, marker = "o", mfc = 'w', ms = 6.5, mec = 'r', mew = 2.5)
line = axes.plot(chi_dn_0_x, chi_dn_0_y)
plt.setp(line, ls = "", c = 'b', lw = 3, marker = "o", mfc = 'w', ms = 6.5, mec = 'b', mew = 2.5)

line = axes.plot(chi_up_roots, chi_dn_roots)
plt.setp(line, ls = "", c = 'k', lw = 3, marker = "o", mfc = 'k', ms = 6.5, mec = 'k', mew = 2.5)


axes.locator_params(axis = 'x', nbins = 6)
axes.locator_params(axis = 'y', nbins = 6)
axes.set_xlabel(r"$\chi_{\rm \uparrow}$", labelpad = 8)
axes.set_ylabel(r"$\chi_{\rm \downarrow}$", labelpad = 8)

fig.savefig("chi_up_dn_la=" + str(la) + "_J="+ str(J) + "_T="+ str(T) + "_B="+ str(B) + "_kx="+ str(len(kx)) +".png")


plt.show()
