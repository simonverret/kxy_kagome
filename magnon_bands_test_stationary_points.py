# -*- coding: Latin-1 -*-

## Modules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
import numpy as np
from numpy import cos, sin, pi, sqrt, exp
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
from functools import partial
from scipy import integrate, optimize
from numpy.linalg import multi_dot
from lmfit import minimize, Parameters, fit_report
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

## Universal Constant :::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

# hbar = 1.05457173e-34 # J.s
# e = 1.60217657e-19 # coulombs
# kB = 1.380648e-23 # J / K

## Variables ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

B = 0.01 # magnetic field in unit of energy g * muB * B

T = 1
kB = 1

J = kB * T
D = 0.2 * J

la_ini = 3.386836
chi_up_ini = -0.599199
chi_dn_ini = -0.583489

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

# Contains the hexago volume in a rectangle of side pi x (2*pi/sqrt(3))
kx = np.linspace(-pi / 3, 2 * pi / 3, 10)
ky = np.linspace(-pi / sqrt(3), pi / sqrt(3), 10)

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

## Full Hamiltonian
def hamiltonian(kx, ky, la, s, B, ts):

    eta1 = - np.array([1, sqrt(3)]) / 2
    eta2 = np.array([1, 0])
    eta3 = np.array([-1, sqrt(3)]) / 2

    k = np.array([kx, ky])

    k1 = np.dot(k, eta1)
    k2 = np.dot(k, eta2)
    k3 = np.dot(k, eta3)

    c1 = cos(k1)
    c2 = cos(k2)
    c3 = cos(k3)

    diag_matrix = (la - s * B) * np.identity(3, dtype = complex)

    tsc = np.conj(ts)

    nondiag_matrix = np.array([[0, ts * c1, tsc * c3],
                               [tsc * c1, 0, ts * c2],
                               [ts * c3, tsc * c2, 0]])

    Hks =  diag_matrix + nondiag_matrix

    return Hks

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

## Diaganolization function
def diag_func(kx, ky, la, s, B, ts):

    """
    (i,j) -> (kx, ky)
    n -> different eigenvalues E[n]
    l -> different components of eigenvectors V[:, n]
    """

    Enks = np.zeros((len(kx), len(ky), 3), dtype = float) # dim: i, j, n
    Vnks = np.zeros((len(kx), len(ky), 3, 3), dtype = complex) # dim: i, j, l, n

    for i in range(len(kx)):
        for j in range(len(ky)):

            Enks[i, j, :], Vnks[i, j, :, :] = np.linalg.eigh(hamiltonian(kx[i], ky[j], la, s, B, ts))
            # The column V[:, n] is the normalized eigenvector corresponding to the eigenvalue E[n]

    # Eigen values of non-diagonal part of the hamiltonian
    Enks_ndiag = Enks - ( la - s * B )

    # Compute la_min for all Enks > 0
    la_min = s * B - np.min(Enks_ndiag)

    return Enks, Enks_ndiag, Vnks, la_min

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

## Hopping term
def compute_ts(chi_up, chi_dn, J, D, s):

    if s == 1:
        return J * ( chi_up + chi_dn ) - 1j * s * D * chi_dn
    else:
        return J * ( chi_up + chi_dn ) - 1j * s * D * chi_up


ts_up_ini = compute_ts(chi_up_ini, chi_dn_ini, J, D, 1)
ts_dn_ini = compute_ts(chi_up_ini, chi_dn_ini, J, D, -1)
print("ts_up_ini = " + str(ts_up_ini))
print("ts_dn_ini = " + str(ts_dn_ini))

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

## Bose - Einstein statics
def n_B(x):

    index_pos = x > 0
    index_neg = x <= 0

    nB = np.zeros(np.shape(x))

    nB[index_pos] = 1 / (exp(x[index_pos]) - 1)
    nB[index_neg] = 0

    return nB

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

## Compute S = sum(Enks/kB*T)
def compute_S(Enks_up, Enks_dn, T):

    Nt = np.shape(Enks_up)[0] * np.shape(Enks_up)[1]

    sum_s_up = np.sum(n_B(Enks_up[:,:,0] / (kB * T))) + np.sum(n_B(Enks_up[:,:,1] / (kB * T))) + np.sum(n_B(Enks_up[:,:,2] / (kB * T)))
    sum_s_dn = np.sum(n_B(Enks_dn[:,:,0] / (kB * T))) + np.sum(n_B(Enks_dn[:,:,1] / (kB * T))) + np.sum(n_B(Enks_dn[:,:,2] / (kB * T)))

    S = (sum_s_up + sum_s_dn) / Nt

    return S

def compute_chi(Enks_up, Enks_dn, Enks_ndiag_up, Enks_ndiag_dn, ts_up, ts_dn, T):

    Nt = np.shape(Enks_up)[0] * np.shape(Enks_up)[1]

    sum_s_up = np.sum(Enks_ndiag_up[:,:,0] * n_B(Enks_up[:,:,0] / (kB * T))) \
             + np.sum(Enks_ndiag_up[:,:,1] * n_B(Enks_up[:,:,1] / (kB * T))) \
             + np.sum(Enks_ndiag_up[:,:,2] * n_B(Enks_up[:,:,2] / (kB * T)))

    sum_s_dn = np.sum(Enks_ndiag_dn[:,:,0] * n_B(Enks_dn[:,:,0] / (kB * T))) \
             + np.sum(Enks_ndiag_dn[:,:,1] * n_B(Enks_dn[:,:,1] / (kB * T))) \
             + np.sum(Enks_ndiag_dn[:,:,2] * n_B(Enks_dn[:,:,2] / (kB * T)))

    chi_up = sum_s_up / Nt / np.absolute(ts_up)
    chi_dn = sum_s_dn / Nt / np.absolute(ts_dn)

    return (chi_up, chi_dn)

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#


#::: Initialize ::::::#

# Compute ts_up & ts_dn
ts_up_ini = compute_ts(chi_up_ini, chi_dn_ini, J, D, 1)
ts_dn_ini = compute_ts(chi_up_ini, chi_dn_ini, J, D, -1)

# Compute eigenvalues
Enks_up_ini, Enks_ndiag_up_ini = diag_func(kx, ky, la_ini, s = 1, B = B, ts = ts_up_ini)[0:2]
Enks_dn_ini, Enks_ndiag_dn_ini = diag_func(kx, ky, la_ini, s = -1, B = B, ts = ts_dn_ini)[0:2]

# Compute residual_S
S_ini = compute_S(Enks_up_ini, Enks_dn_ini, T)

#::::::::: New ::::::#

# Compute residual_Chi
(chi_up_new, chi_dn_new) = compute_chi(Enks_up_ini, Enks_dn_ini, Enks_ndiag_up_ini, Enks_ndiag_dn_ini, ts_up_ini, ts_dn_ini, T)

print("IN (lambda, chi_up, chi_dn) = ("
       + r"{0:.4e}".format(la_ini) + ", "
       + r"{0:.4e}".format(chi_up_ini) + ", "
       + r"{0:.4e}".format(chi_dn_ini) + ")")


print("OUT (S, chi_up, chi_dn) = ("
       + r"{0:.4e}".format(S_ini) + ", "
       + r"{0:.4e}".format(chi_up_new) + ", "
       + r"{0:.4e}".format(chi_dn_new) + ")")

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


# ## S vs lambda ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

# fig , axes = plt.subplots(1,1, figsize=(9.2, 5.6)) # figsize is w x h in inch of figure
# fig.subplots_adjust(left = 0.17, right = 0.81, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

# for tick in axes.xaxis.get_major_ticks():
#     tick.set_pad(7)
# for tick in axes.yaxis.get_major_ticks():
#     tick.set_pad(8)

# axes.axhline(y = 0.5, ls = "--", c = "k", linewidth = 0.6)

# #///// Plot /////#

# la_array = np.arange(la*0.9, la*1.1, 0.01)
# S_array = np.zeros(len(la_array))

# for i, laa in enumerate(la_array):
#     Enks_up, Enks_ndiag_up, Vnks_up, la_min_up = diag_func(kx, ky, laa, s = 1, B = B, ts = ts_up)
#     Enks_dn, Enks_ndiag_dn, Vnks_dn, la_min_dn = diag_func(kx, ky, laa, s = -1, B = B, ts = ts_dn)
#     S_array[i] = compute_S(Enks_up, Enks_dn, T)

# line = axes.plot(la_array, S_array)
# plt.setp(line, ls = "--", c = '#F60000', lw = 3, marker = "o", mfc = 'w', ms = 6.5, mec = '#F60000', mew = 2.5)

# axes.locator_params(axis = 'x', nbins = 6)
# axes.locator_params(axis = 'y', nbins = 6)
# axes.set_xlabel(r"$\lambda$", labelpad = 8)
# axes.set_ylabel(r"$S$", labelpad = 8)



# ## Berry phase vs kx ky = 0 :::::::::::::::::::::::::::::::::::::::::::::::::::#

# fig , axes = plt.subplots(1,1, figsize=(9.2, 5.6)) # figsize is w x h in inch of figure
# fig.subplots_adjust(left = 0.17, right = 0.81, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

# for tick in axes.xaxis.get_major_ticks():
#     tick.set_pad(7)
# for tick in axes.yaxis.get_major_ticks():
#     tick.set_pad(8)

# #///// Plot /////#

# Omega_nks = berry_phase(kx, np.zeros(1), la, s, B, ts)

# line = axes.plot(kx, Omega_nks[:,0, 0])
# plt.setp(line, ls = "-", c = '#FF0000', lw = 3, marker = "", mfc = 'w', ms = 6.5, mec = '#FF0000', mew = 2.5)
# line = axes.plot(kx, Omega_nks[:,0, 1])
# plt.setp(line, ls = "-", c = '#00E054', lw = 3, marker = "", mfc = 'w', ms = 6.5, mec = '#00E054', mew = 2.5)
# line = axes.plot(kx, Omega_nks[:,0, 2])
# plt.setp(line, ls = "-", c = '#7D44FF', lw = 3, marker = "", mfc = 'w', ms = 6.5, mec = '#7D44FF', mew = 2.5)

# axes.locator_params(axis = 'x', nbins = 6)
# axes.locator_params(axis = 'y', nbins = 6)
# axes.set_xlabel(r"$k_{\rm x}$", labelpad = 8)
# axes.set_ylabel(r"$\Omega$", labelpad = 8)



# ## Energy vs kx ky = 0:::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

# fig , axes = plt.subplots(1,1, figsize=(9.2, 5.6)) # figsize is w x h in inch of figure
# fig.subplots_adjust(left = 0.17, right = 0.81, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

# for tick in axes.xaxis.get_major_ticks():
#     tick.set_pad(7)
# for tick in axes.yaxis.get_major_ticks():
#     tick.set_pad(8)

# #///// Plot /////#

# Enks_up, Enks_ndiag_up, Vnks_up, la_min_up = diag_func(kx, np.zeros(1), la, s = 1, B = B, ts = ts_up)

# line = axes.plot(kx, Enks_up[:,0, 0])
# plt.setp(line, ls = "-", c = '#FF0000', lw = 3, marker = "", mfc = 'w', ms = 6.5, mec = '#FF0000', mew = 2.5)
# line = axes.plot(kx, Enks_up[:,0, 1])
# plt.setp(line, ls = "-", c = '#00E054', lw = 3, marker = "", mfc = 'w', ms = 6.5, mec = '#00E054', mew = 2.5)
# line = axes.plot(kx, Enks_up[:,0, 2])
# plt.setp(line, ls = "-", c = '#7D44FF', lw = 3, marker = "", mfc = 'w', ms = 6.5, mec = '#7D44FF', mew = 2.5)

# Enks_dn, Enks_ndiag_dn, Vnks_dn, la_min_dn = diag_func(kx, np.zeros(1), la, s = 1, B = B, ts = ts_dn)

# line = axes.plot(kx, Enks_dn[:,0, 0])
# plt.setp(line, ls = "--", c = '#FF0000', lw = 3, marker = "", mfc = 'w', ms = 6.5, mec = '#FF0000', mew = 2.5)
# line = axes.plot(kx, Enks_dn[:,0, 1])
# plt.setp(line, ls = "--", c = '#00E054', lw = 3, marker = "", mfc = 'w', ms = 6.5, mec = '#00E054', mew = 2.5)
# line = axes.plot(kx, Enks_dn[:,0, 2])
# plt.setp(line, ls = "--", c = '#7D44FF', lw = 3, marker = "", mfc = 'w', ms = 6.5, mec = '#7D44FF', mew = 2.5)

# axes.set_xlim(0, 2*np.pi/3)
# axes.locator_params(axis = 'x', nbins = 6)
# axes.locator_params(axis = 'y', nbins = 6)
# axes.set_xlabel(r"$k_{\rm x}$", labelpad = 8)
# axes.set_ylabel(r"$E$", labelpad = 8)



# ## Energy bands :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

# fig = plt.figure(figsize=(9.2, 5.6))
# axes = fig.add_subplot(111, projection='3d')

# Enks, Enks_ndiag, Vnks, la_min = diag_func(kx, ky, la, s = 1, B = B, ts = ts_up)

# kxx, kyy = np.meshgrid(kx, ky, indexing = 'xy')
# axes.plot_surface(kxx, kyy, Enks[:, :, 0], rstride=1, cstride=1, alpha=1, color = "#FF0000")
# axes.plot_surface(kxx, kyy, Enks[:, :, 1], rstride=1, cstride=1, alpha=1, color = "#00E054")
# axes.plot_surface(kxx, kyy, Enks[:, :, 2], rstride=1, cstride=1, alpha=1, color = "#7D44FF")

# axes.set_xlim3d(-np.pi, np.pi)
# axes.set_ylim3d(-np.pi, np.pi)
# # axes.set_zlim3d(bottom = 0)

# axes.set_xlabel(r"$k_{\rm x}$", labelpad = 20)
# axes.set_ylabel(r"$k_{\rm y}$", labelpad = 20)
# axes.set_zlabel(r"$E$", labelpad = 20)

# ## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#




## diff_chi vs chi ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
la = 2.5

# span = 0.01
# # la = 3
# chi_up_array = np.arange(-2, 2, 0.01)
# chi_dn_array = np.arange(-2, 2, 0.01)

# ts_up_array = np.zeros(np.shape(chi_up_array))
# ts_dn_array = np.zeros(np.shape(chi_dn_array))

# diff_chi_up = np.zeros((len(chi_up_array), len(chi_dn_array)), dtype = float)
# diff_chi_dn = np.zeros((len(chi_dn_array), len(chi_dn_array)), dtype = float)

# chi_up_0_x = []
# chi_up_0_y = []
# chi_dn_0_x = []
# chi_dn_0_y = []

# for i, chi_up in enumerate(chi_up_array):
#     for j, chi_dn in enumerate(chi_dn_array):

#         ts_up = compute_ts(chi_up, chi_dn, J, D, 1)
#         ts_dn = compute_ts(chi_up, chi_dn, J, D, -1)

#         Enks_up, Enks_ndiag_up, Vnks_up, la_min_up = diag_func(kx, ky, la, s = 1, B = B, ts = ts_up)
#         Enks_dn, Enks_ndiag_dn, Vnks_dn, la_min_dn = diag_func(kx, ky, la, s = -1, B = B, ts = ts_dn)

#         (chi_up_new, chi_dn_new) = compute_chi(Enks_up, Enks_dn, Enks_ndiag_up, Enks_ndiag_dn, ts_up, ts_dn, T)

#         diff_chi_up[i, j] = chi_up_new - chi_up
#         diff_chi_dn[i, j] = chi_dn_new - chi_dn

#         if (diff_chi_up[i, j] < span) * (diff_chi_up[i, j] > -span):
#             chi_up_0_x.append(chi_up)
#             chi_up_0_y.append(chi_dn)

#         if (diff_chi_dn[i, j] < span) * (diff_chi_dn[i, j] > -span):
#             chi_dn_0_x.append(chi_up)
#             chi_dn_0_y.append(chi_dn)





# ## chi for diff_chi = 0 :::::::::::::::::::::::::::::::::::::::::::::::::::::#

# fig , axes = plt.subplots(1,1, figsize=(9.2, 5.6)) # figsize is w x h in inch of figure
# fig.subplots_adjust(left = 0.17, right = 0.81, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

# for tick in axes.xaxis.get_major_ticks():
#     tick.set_pad(7)
# for tick in axes.yaxis.get_major_ticks():
#     tick.set_pad(8)

# #///// Plot /////#

# line = axes.plot(chi_up_0_x, chi_up_0_y)
# plt.setp(line, ls = "", c = 'r', lw = 3, marker = "o", mfc = 'w', ms = 6.5, mec = '#FF0000', mew = 2.5)
# line = axes.plot(chi_dn_0_x, chi_dn_0_y)
# plt.setp(line, ls = "", c = 'b', lw = 3, marker = "o", mfc = 'w', ms = 6.5, mec = '#00E054', mew = 2.5)

# # axes.set_xlim(0, 2*np.pi/3)
# axes.locator_params(axis = 'x', nbins = 6)
# axes.locator_params(axis = 'y', nbins = 6)
# axes.set_xlabel(r"$\chi_{\rm \uparrow}$", labelpad = 8)
# axes.set_ylabel(r"$\chi_{\rm \downarrow}$", labelpad = 8)


# ## 3D diff_chi vs chi :::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

# fig = plt.figure(figsize=(9.2, 5.6))
# axes = fig.add_subplot(111, projection='3d')

# # ij_index_up = (diff_chi_up > 0.05)
# # ij_index_dn = (diff_chi_dn > 0.05)

# # diff_chi_up[ij_index_up] = np.nan
# # diff_chi_dn[ij_index_dn] = np.nan

# # ij_index_up = (diff_chi_up < -0.05)
# # ij_index_dn = (diff_chi_dn < -0.05)

# # diff_chi_up[ij_index_up] = np.nan
# # diff_chi_dn[ij_index_dn] = np.nan


# chii_up, chii_dn = np.meshgrid(chi_up_array, chi_dn_array, indexing = 'xy')
# axes.scatter(chii_up, chii_dn, diff_chi_up, color = "#FF5555")
# axes.scatter(chii_up, chii_dn, diff_chi_dn, color = "#511CFF")
# # axes.plot_surface(chii_up, chii_dn, np.zeros((len(chi_up_array), len(chi_dn_array)), dtype = float), rstride=1, cstride=1, alpha=1, color = "k")


# # axes.set_xlim3d(-np.pi, np.pi)
# # axes.set_ylim3d(-np.pi, np.pi)
# axes.set_zlim3d(-1, 1)

# axes.set_xlabel(r"$\chi_{\rm \uparrow}$", labelpad = 20)
# axes.set_ylabel(r"$\chi_{\rm \downarrow}$", labelpad = 20)
# axes.set_zlabel(r"$\Delta\chi$", labelpad = 20)

# ## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#





# ## residual_chi vs chi ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

# ## chi_dn residual function
# def residual_chi_dn(pars, chi_up, la, kx, ky, B, J, D, T):

#     chi_dn = pars["chi_dn"].value

#     # Compute ts_up & ts_dn
#     ts_up = compute_ts(chi_up, chi_dn, J, D, 1)
#     ts_dn = compute_ts(chi_up, chi_dn, J, D, -1)

#     # Compute eigenvalues
#     Enks_up, Enks_ndiag_up = diag_func(kx, ky, la, s = 1, B = B, ts = ts_up)[0:2]
#     Enks_dn, Enks_ndiag_dn = diag_func(kx, ky, la, s = -1, B = B, ts = ts_dn)[0:2]

#     # Compute residual_Chi
#     (chi_up_new, chi_dn_new) = compute_chi(Enks_up, Enks_dn, Enks_ndiag_up, Enks_ndiag_dn, ts_up, ts_dn, T)
#     residual_chi_dn = chi_dn - chi_dn_new

#     print("residual(chi_dn) = ("
#        + r"{0:.4e}".format(residual_chi_dn) + ")")

#     return residual_chi_dn

# ## chi_dn residual function
# def residual_chi_up(pars, chi_dn, la, kx, ky, B, J, D, T):

#     chi_up = pars["chi_up"].value

#     # Compute ts_up & ts_dn
#     ts_up = compute_ts(chi_up, chi_dn, J, D, 1)
#     ts_dn = compute_ts(chi_up, chi_dn, J, D, -1)

#     # Compute eigenvalues
#     Enks_up, Enks_ndiag_up = diag_func(kx, ky, la, s = 1, B = B, ts = ts_up)[0:2]
#     Enks_dn, Enks_ndiag_dn = diag_func(kx, ky, la, s = -1, B = B, ts = ts_dn)[0:2]

#     # Compute residual_Chi
#     (chi_up_new, chi_dn_new) = compute_chi(Enks_up, Enks_dn, Enks_ndiag_up, Enks_ndiag_dn, ts_up, ts_dn, T)
#     residual_chi_up = chi_up - chi_up_new

#     print("residual(chi_up) = ("
#        + r"{0:.4e}".format(residual_chi_up) + ")")

#     return residual_chi_up





# chi_up_array = np.arange(-0.5, -0.1, 0.01)
# chi_dn_array = np.arange(-0.5, -0.1, 0.01)

# ts_up_array = np.zeros(np.shape(chi_up_array))
# ts_dn_array = np.zeros(np.shape(chi_dn_array))

# chi_up_0_x = []
# chi_up_0_y = []
# chi_dn_0_x = []
# chi_dn_0_y = []

# for i, chi_up in enumerate(chi_up_array):


#     parameters = Parameters()
#     parameters.add("chi_dn", value = -0.2, min = -0.7, max = 0, vary = True)
#     # parameters.add("chi_dn", min = -0.5, max = -0.1, brute_step=0.001)
#     out = minimize(residual_chi_dn, parameters, args=(chi_up, la, kx, ky, B, J, D, T), method="least_squares")

#     chi_up_0_x.append(chi_up)
#     chi_up_0_y.append(out.params["chi_dn"].value)

# for i, chi_dn in enumerate(chi_dn_array):

#         parameters = Parameters()
#         parameters.add("chi_up", value = -0.2, min = -0.7, max = 0, vary = True)
#         # parameters.add("chi_up", min = -0.5, max = -0.1, brute_step=0.001)
#         out = minimize(residual_chi_up, parameters, args=(chi_dn, la, kx, ky, B, J, D, T), method="least_squares")

#         chi_dn_0_x.append(out.params["chi_up"].value)
#         chi_dn_0_y.append(chi_dn)




# ## chi for diff_chi = 0 :::::::::::::::::::::::::::::::::::::::::::::::::::::::#

# fig , axes = plt.subplots(1,1, figsize=(9.2, 5.6)) # figsize is w x h in inch of figure
# fig.subplots_adjust(left = 0.17, right = 0.81, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

# for tick in axes.xaxis.get_major_ticks():
#     tick.set_pad(7)
# for tick in axes.yaxis.get_major_ticks():
#     tick.set_pad(8)

# #///// Plot /////#

# line = axes.plot(chi_up_0_x, chi_up_0_y)
# plt.setp(line, ls = "", c = 'r', lw = 3, marker = "o", mfc = 'w', ms = 6.5, mec = 'r', mew = 2.5)
# line = axes.plot(chi_dn_0_x, chi_dn_0_y)
# plt.setp(line, ls = "", c = 'b', lw = 3, marker = "o", mfc = 'w', ms = 6.5, mec = 'b', mew = 2.5)

# # axes.set_xlim(0, 2*np.pi/3)
# axes.locator_params(axis = 'x', nbins = 6)
# axes.locator_params(axis = 'y', nbins = 6)
# axes.set_xlabel(r"$\chi_{\rm \uparrow}$", labelpad = 8)
# axes.set_ylabel(r"$\chi_{\rm \downarrow}$", labelpad = 8)








plt.show()














## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

# ## Derivative dH/dkx
# def derivative_kx_hamiltonian(kx, ky, la, s, B, ts):

#     eta1 = - np.array([1, sqrt(3)]) / 2
#     eta2 = np.array([1, 0])
#     eta3 = np.array([-1, sqrt(3)]) / 2

#     k = np.array([kx, ky])

#     k1 = np.dot(k, eta1)
#     k2 = np.dot(k, eta2)
#     k3 = np.dot(k, eta3)

#     s1 = 0.5 * sin(k1)
#     s2 = - sin(k2)
#     s3 = 0.5 * sin(k3)

#     tsc = np.conj(ts)

#     dHksdkx = np.array([[0, ts * s1, tsc * s3],
#                                [tsc * s1, 0, ts * s2],
#                                [ts * s3, tsc * s2, 0]])



#     return dHksdkx

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

# ## Derivative dH/dkx
# def derivative_ky_hamiltonian(kx, ky, la, s, B, ts):

#     eta1 = - np.array([1, sqrt(3)]) / 2
#     eta2 = np.array([1, 0])
#     eta3 = np.array([-1, sqrt(3)]) / 2

#     k = np.array([kx, ky])

#     k1 = np.dot(k, eta1)
#     k2 = np.dot(k, eta2)
#     k3 = np.dot(k, eta3)

#     s1 = sqrt(3) / 2 * sin(k1)
#     s2 = 0 * sin(k2)
#     s3 = - sqrt(3) / 2 * sin(k3)

#     tsc = np.conj(ts)

#     dHksdky = np.array([[0, ts * s1, tsc * s3],
#                         [tsc * s1, 0, ts * s2],
#                         [ts * s3, tsc * s2, 0]])



#     return dHksdky

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

# ## Kronecker Delta
# def kronecker(n,m):

#     if n == m: return 1
#     else: return 0

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

# ## Berry phase

# def berry_phase(kx, ky, la, s, B, ts):

#     """
#     (i,j) -> (kx, ky)
#     n -> different eigenvalues E[n]
#     l -> different components of eigenvectors V[:, n]
#     """

#     Enks = np.zeros((len(kx), len(ky), 3), dtype = float) # dim: i, j, n
#     Vnks = np.zeros((len(kx), len(ky), 3, 3), dtype = complex) # dim: i, j, l, n

#     dHdkx = np.zeros((len(kx), len(ky), 3, 3), dtype = complex) # dim: i, j, n x n (matrix dim)
#     dHdky = np.zeros((len(kx), len(ky), 3, 3), dtype = complex) # dim: i, j, n x n (matrix dim)

#     dVdkx = np.zeros((len(kx), len(ky), 3, 3), dtype = complex) # # dim: i, j, l, n
#     dVdky = np.zeros((len(kx), len(ky), 3, 3), dtype = complex) # # dim: i, j, l, n

#     Omega_nks = np.zeros((len(kx), len(ky), 3), dtype = float) # dim: i, j, n

#     for i in range(len(kx)):
#         for j in range(len(ky)):

#             Enks[i, j, :], Vnks[i, j, :, :] = np.linalg.eigh(hamiltonian(kx[i], ky[j], la, s, B, ts))
#             # The column V[:, n] is the normalized eigenvector corresponding to the eigenvalue E[n]

#             dHdkx[i, j, :, :] = derivative_kx_hamiltonian(kx[i], ky[j], la, s, B, ts)
#             dHdky[i, j, :, :] = derivative_ky_hamiltonian(kx[i], ky[j], la, s, B, ts)

#             dVdkx[i,j,:,0] = multi_dot([Vnks[i,j,:,1], dHdkx[i, j, :, :], Vnks[i,j,:,0]]) / (Enks[i, j, 0] - Enks[i, j, 1]) * Vnks[i,j,:,1]  \
#                            + multi_dot([Vnks[i,j,:,2], dHdkx[i, j, :, :], Vnks[i,j,:,0]]) / (Enks[i, j, 0] - Enks[i, j, 2]) * Vnks[i,j,:,2]
#             dVdkx[i,j,:,1] = multi_dot([Vnks[i,j,:,0], dHdkx[i, j, :, :], Vnks[i,j,:,1]]) / (Enks[i, j, 1] - Enks[i, j, 0]) * Vnks[i,j,:,0]  \
#                            + multi_dot([Vnks[i,j,:,2], dHdkx[i, j, :, :], Vnks[i,j,:,1]]) / (Enks[i, j, 1] - Enks[i, j, 2]) * Vnks[i,j,:,2]
#             dVdkx[i,j,:,2] = multi_dot([Vnks[i,j,:,0], dHdkx[i, j, :, :], Vnks[i,j,:,2]]) / (Enks[i, j, 2] - Enks[i, j, 0]) * Vnks[i,j,:,0]  \
#                            + multi_dot([Vnks[i,j,:,1], dHdkx[i, j, :, :], Vnks[i,j,:,2]]) / (Enks[i, j, 2] - Enks[i, j, 1]) * Vnks[i,j,:,1]

#             dVdky[i,j,:,0] = multi_dot([Vnks[i,j,:,1], dHdky[i, j, :, :], Vnks[i,j,:,0]]) / (Enks[i, j, 0] - Enks[i, j, 1]) * Vnks[i,j,:,1]  \
#                            + multi_dot([Vnks[i,j,:,2], dHdky[i, j, :, :], Vnks[i,j,:,0]]) / (Enks[i, j, 0] - Enks[i, j, 2]) * Vnks[i,j,:,2]
#             dVdky[i,j,:,1] = multi_dot([Vnks[i,j,:,0], dHdky[i, j, :, :], Vnks[i,j,:,1]]) / (Enks[i, j, 1] - Enks[i, j, 0]) * Vnks[i,j,:,0]  \
#                            + multi_dot([Vnks[i,j,:,2], dHdky[i, j, :, :], Vnks[i,j,:,1]]) / (Enks[i, j, 1] - Enks[i, j, 2]) * Vnks[i,j,:,2]
#             dVdky[i,j,:,2] = multi_dot([Vnks[i,j,:,0], dHdky[i, j, :, :], Vnks[i,j,:,2]]) / (Enks[i, j, 2] - Enks[i, j, 0]) * Vnks[i,j,:,0]  \
#                            + multi_dot([Vnks[i,j,:,1], dHdky[i, j, :, :], Vnks[i,j,:,2]]) / (Enks[i, j, 2] - Enks[i, j, 1]) * Vnks[i,j,:,1]


#             Omega_nks[i,j,0] = 2 * np.real( 1j * np.dot(dVdkx[i,j,:,0],dVdky[i,j,:,0]))
#             Omega_nks[i,j,1] = 2 * np.real( 1j * np.dot(dVdkx[i,j,:,1],dVdky[i,j,:,1]))
#             Omega_nks[i,j,2] = 2 * np.real( 1j * np.dot(dVdkx[i,j,:,2],dVdky[i,j,:,2]))

#     return Omega_nks

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
