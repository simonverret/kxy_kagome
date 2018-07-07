# -*- coding: Latin-1 -*-

## Modules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
import numpy as np
from numpy import cos, sin, pi, sqrt, exp
np.set_printoptions(6,suppress=True,sign="+",floatmode="fixed")
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
kB = 1

## Variables ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

B = 0.01 # magnetic field in unit of energy g * muB * B

T = 0.8

J = 1
D = 0.2 * J

chi_up_ini = -1
chi_dn_ini = -1

resolution_k = 33

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

# Contains the hexago volume in a rectangle of side pi x (2*pi/sqrt(3))
kx = np.linspace(-pi / 3, 2 * pi / 3, resolution_k)
ky = np.linspace(-pi / sqrt(3), pi / sqrt(3), resolution_k)
kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#


## Initialize guessed values for root function ##

ts_up_ini = bands_func.compute_ts(chi_up_ini, chi_dn_ini, J, D, 1)
ts_dn_ini = bands_func.compute_ts(chi_up_ini, chi_dn_ini, J, D, -1)

la_min_up = bands_func.diag_func(kxx, kyy, la = 0, s = 1, B = B, ts = ts_up_ini)[-1]
la_min_dn = bands_func.diag_func(kxx, kyy, la = 0, s = -1, B = B, ts = ts_dn_ini)[-1]

la_min = np.max([la_min_up, la_min_dn]) # prevent from having negative eigen values in the root algorithm

p_residual_S_chi = partial(bands_func.residual_S_chi, kx = kxx, ky = kyy, B = B, D = D, J = J, T = T)

# In order to avoid the trivial value for (chi_up, chi_dn) = (0,0), we look for
# chi roots different from the trivial ones by trying different chi_ini values
# starting from chi_ini ~ 0 to higher values, as the non trivial roots are the second
# roots to find before chi_function becomes discontinous:

chi_steps = np.arange(0.41, 5, 0.2)
for chi_ini in chi_steps:

        out = optimize.root(p_residual_S_chi, np.array([la_min, -chi_ini, -chi_ini]))
        roots = out.x

        if np.all(np.abs(roots[1:]) < 1e-4) or (out.success is False) : # (chi_up, chi_dn) < 1e-4
            continue
        else:
            break

print(roots)

la = roots[0]
chi_up = roots[1]
chi_dn = roots[2]

## Compute bands from the right lambda, chi_up and chi_dn >>>>>>>>>>>>>>>>>>>>>#
ts_up = bands_func.compute_ts(chi_up, chi_dn, J, D, 1)
ts_dn = bands_func.compute_ts(chi_up, chi_dn, J, D, -1)

Enks_up, Enks_ndiag_up, Vnks_up, dHks_dkx_up, dHks_dky_up = bands_func.diag_func(kxx, kyy, la, s = 1, B = B, ts = ts_up)[0:-1]
Enks_dn, Enks_ndiag_dn, Vnks_dn, dHks_dkx_dn, dHks_dky_dn = bands_func.diag_func(kxx, kyy, la, s = -1, B = B, ts = ts_dn)[0:-1]

Omega_nks_up = bands_func.berry_phase(Enks_up, Vnks_up, dHks_dkx_up, dHks_dky_up)
Omega_nks_dn = bands_func.berry_phase(Enks_dn, Vnks_dn, dHks_dkx_dn, dHks_dky_dn)

kxy = bands_func.kxy_func(Enks_up, Enks_dn, Omega_nks_up, Omega_nks_dn, T)

print("kxy = " + str(kxy))

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


## S vs lambda ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

fig , axes = plt.subplots(1,1, figsize=(9.2, 5.6)) # figsize is w x h in inch of figure
fig.subplots_adjust(left = 0.17, right = 0.81, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

for tick in axes.xaxis.get_major_ticks():
    tick.set_pad(7)
for tick in axes.yaxis.get_major_ticks():
    tick.set_pad(8)

axes.axhline(y = 0.5, ls = "--", c = "k", linewidth = 0.6)

#///// Plot /////#

la_array = np.linspace(la*0.9, la*1.1, 10)
S_array = np.zeros(len(la_array))

for i, laa in enumerate(la_array):
    Enks_up = bands_func.diag_func(kxx, kyy, laa, s = 1, B = B, ts = ts_up)[0]
    Enks_dn = bands_func.diag_func(kxx, kyy, laa, s = -1, B = B, ts = ts_dn)[0]
    S_array[i] = bands_func.compute_S(Enks_up, Enks_dn, T)

line = axes.plot(la_array, S_array)
plt.setp(line, ls = "--", c = '#F60000', lw = 3, marker = "o", mfc = 'w', ms = 6.5, mec = '#F60000', mew = 2.5)

axes.locator_params(axis = 'x', nbins = 6)
axes.locator_params(axis = 'y', nbins = 6)
axes.set_xlabel(r"$\lambda$", labelpad = 8)
axes.set_ylabel(r"$S$", labelpad = 8)




## Special meshgrid for 2D plot (_p for "plot") :::::::::::::::::::::::::::::::#

kxx_p, kyy_p = np.meshgrid(kx, np.zeros(1), indexing = 'ij')

Enks_up_p, Enks_ndiag_up_p, Vnks_up_p, dHks_dkx_up_p, dHks_dky_up_p = bands_func.diag_func(kxx_p, kyy_p, la, s = 1, B = B, ts = ts_up)[0:-1]
Enks_dn_p, Enks_ndiag_dn_p, Vnks_dn_p, dHks_dkx_dn_p, dHks_dky_dn_p = bands_func.diag_func(kxx_p, kyy_p, la, s = -1, B = B, ts = ts_dn)[0:-1]

Omega_nks_up_p = bands_func.berry_phase(Enks_up_p, Vnks_up_p, dHks_dkx_up_p, dHks_dky_up_p)
Omega_nks_dn_p = bands_func.berry_phase(Enks_dn_p, Vnks_dn_p, dHks_dkx_dn_p, dHks_dky_dn_p)

## Berry phase vs kx ky = 0 :::::::::::::::::::::::::::::::::::::::::::::::::::#

fig , axes = plt.subplots(1,1, figsize=(9.2, 5.6)) # figsize is w x h in inch of figure
fig.subplots_adjust(left = 0.17, right = 0.81, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

for tick in axes.xaxis.get_major_ticks():
    tick.set_pad(7)
for tick in axes.yaxis.get_major_ticks():
    tick.set_pad(8)

#///// Plot /////#

line = axes.plot(kxx_p / pi, Omega_nks_up_p[:,0, 0])
plt.setp(line, ls = "-", c = '#FF0000', lw = 3, marker = "", mfc = 'w', ms = 6.5, mec = '#FF0000', mew = 2.5)
line = axes.plot(kxx_p / pi, Omega_nks_up_p[:,0, 1])
plt.setp(line, ls = "-", c = '#00E054', lw = 3, marker = "", mfc = 'w', ms = 6.5, mec = '#00E054', mew = 2.5)
line = axes.plot(kxx_p / pi, Omega_nks_up_p[:,0, 2])
plt.setp(line, ls = "-", c = '#7D44FF', lw = 3, marker = "", mfc = 'w', ms = 6.5, mec = '#7D44FF', mew = 2.5)

# line = axes.plot(kxx_p / pi, Omega_nks_dn_p[:,0, 0])
# plt.setp(line, ls = "--", c = '#FF0000', lw = 3, marker = "", mfc = 'w', ms = 6.5, mec = '#FF0000', mew = 2.5)
# line = axes.plot(kxx_p / pi, Omega_nks_dn_p[:,0, 1])
# plt.setp(line, ls = "--", c = '#00E054', lw = 3, marker = "", mfc = 'w', ms = 6.5, mec = '#00E054', mew = 2.5)
# line = axes.plot(kxx_p / pi, Omega_nks_dn_p[:,0, 2])
# plt.setp(line, ls = "--", c = '#7D44FF', lw = 3, marker = "", mfc = 'w', ms = 6.5, mec = '#7D44FF', mew = 2.5)

axes.set_xlim(0, 2/3)
axes.locator_params(axis = 'x', nbins = 6)
axes.locator_params(axis = 'y', nbins = 6)
axes.set_xlabel(r"$k_{\rm x}$", labelpad = 8)
axes.set_ylabel(r"$\Omega$", labelpad = 8)



## Energy vs kx ky = 0:::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

fig , axes = plt.subplots(1,1, figsize=(9.2, 5.6)) # figsize is w x h in inch of figure
fig.subplots_adjust(left = 0.17, right = 0.81, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

for tick in axes.xaxis.get_major_ticks():
    tick.set_pad(7)
for tick in axes.yaxis.get_major_ticks():
    tick.set_pad(8)

#///// Plot /////#

Enks_up_p = bands_func.diag_func(kxx_p, kyy_p, la, s = 1, B = B, ts = ts_up)[0]

line = axes.plot(kxx_p / pi, Enks_up_p[:,0, 0])
plt.setp(line, ls = "-", c = '#FF0000', lw = 3, marker = "", mfc = 'w', ms = 6.5, mec = '#FF0000', mew = 2.5)
line = axes.plot(kxx_p / pi, Enks_up_p[:,0, 1])
plt.setp(line, ls = "-", c = '#00E054', lw = 3, marker = "", mfc = 'w', ms = 6.5, mec = '#00E054', mew = 2.5)
line = axes.plot(kxx_p / pi, Enks_up_p[:,0, 2])
plt.setp(line, ls = "-", c = '#7D44FF', lw = 3, marker = "", mfc = 'w', ms = 6.5, mec = '#7D44FF', mew = 2.5)

Enks_dn_p = bands_func.diag_func(kxx_p, kyy_p, la, s = -1, B = B, ts = ts_dn)[0]

line = axes.plot(kxx_p / pi, Enks_dn_p[:,0, 0])
plt.setp(line, ls = "--", c = '#FF0000', lw = 3, marker = "", mfc = 'w', ms = 6.5, mec = '#FF0000', mew = 2.5)
line = axes.plot(kxx_p / pi, Enks_dn_p[:,0, 1])
plt.setp(line, ls = "--", c = '#00E054', lw = 3, marker = "", mfc = 'w', ms = 6.5, mec = '#00E054', mew = 2.5)
line = axes.plot(kxx_p / pi, Enks_dn_p[:,0, 2])
plt.setp(line, ls = "--", c = '#7D44FF', lw = 3, marker = "", mfc = 'w', ms = 6.5, mec = '#7D44FF', mew = 2.5)

axes.set_xlim(0, 2/3)
axes.locator_params(axis = 'x', nbins = 6)
axes.locator_params(axis = 'y', nbins = 6)
axes.set_xlabel(r"$k_{\rm x} / \pi$", labelpad = 8)
axes.set_ylabel(r"$E$", labelpad = 8)



# ## Energy bands :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

# fig = plt.figure(figsize=(9.2, 5.6))
# axes = fig.add_subplot(111, projection='3d')

# axes.plot_surface(kxx / pi, kyy / pi, Enks_up[:, :, 0], rstride=1, cstride=1, alpha=1, color = "#FF0000")
# axes.plot_surface(kxx / pi, kyy / pi, Enks_up[:, :, 1], rstride=1, cstride=1, alpha=1, color = "#00E054")
# axes.plot_surface(kxx / pi, kyy / pi, Enks_up[:, :, 2], rstride=1, cstride=1, alpha=1, color = "#7D44FF")

# axes.plot_surface(kxx / pi, kyy / pi, Enks_dn[:, :, 0], rstride=1, cstride=1, alpha=1, color = "#FF0000")
# axes.plot_surface(kxx / pi, kyy / pi, Enks_dn[:, :, 1], rstride=1, cstride=1, alpha=1, color = "#00E054")
# axes.plot_surface(kxx / pi, kyy / pi, Enks_dn[:, :, 2], rstride=1, cstride=1, alpha=1, color = "#7D44FF")

# axes.set_xlim3d(-2/3, 2/3)
# axes.set_ylim3d(-2/3, 2/3)
# # axes.set_zlim3d(bottom = 0)

# axes.set_xlabel(r"$k_{\rm x} / \pi$", labelpad = 20)
# axes.set_ylabel(r"$k_{\rm y} / \pi$", labelpad = 20)
# axes.set_zlabel(r"$E$", labelpad = 20)

# ## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#





plt.show()








