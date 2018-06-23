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

la_ini = 0
chi_up_ini = -1
chi_dn_ini = -1

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
# print("ts_up_ini = " + str(ts_up_ini))
# print("ts_dn_ini = " + str(ts_dn_ini))

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

def residual_chi(chi, la, kx, ky, B, T):

    chi_up = chi[0]
    chi_dn = chi[1]

    # Compute ts_up & ts_dn
    ts_up = compute_ts(chi_up, chi_dn, J, D, 1)
    ts_dn = compute_ts(chi_up, chi_dn, J, D, -1)

    # Compute eigenvalues
    Enks_up, Enks_ndiag_up = diag_func(kx, ky, la, s = 1, B = B, ts = ts_up)[0:2]
    Enks_dn, Enks_ndiag_dn = diag_func(kx, ky, la, s = -1, B = B, ts = ts_dn)[0:2]

    # Compute residual_Chi
    (chi_up_new, chi_dn_new) = compute_chi(Enks_up, Enks_dn, Enks_ndiag_up, Enks_ndiag_dn, ts_up, ts_dn, T)
    residual_chi_up = chi_up - chi_up_new
    residual_chi_dn = chi_dn - chi_dn_new

    return (residual_chi_up, residual_chi_dn)

def residual_chi_up(chi_up, chi_dn, la, kx, ky, B, T):

    # Compute ts_up & ts_dn
    ts_up = compute_ts(chi_up, chi_dn, J, D, 1)
    ts_dn = compute_ts(chi_up, chi_dn, J, D, -1)

    # Compute eigenvalues
    Enks_up, Enks_ndiag_up = diag_func(kx, ky, la, s = 1, B = B, ts = ts_up)[0:2]
    Enks_dn, Enks_ndiag_dn = diag_func(kx, ky, la, s = -1, B = B, ts = ts_dn)[0:2]

    # Compute residual_Chi
    (chi_up_new, chi_dn_new) = compute_chi(Enks_up, Enks_dn, Enks_ndiag_up, Enks_ndiag_dn, ts_up, ts_dn, T)
    residual_chi_up = chi_up - chi_up_new
    # residual_chi_dn = chi_dn - chi_dn_new

    return (residual_chi_up)

def residual_chi_dn(chi_dn, chi_up, la, kx, ky, B, T):

    # Compute ts_up & ts_dn
    ts_up = compute_ts(chi_up, chi_dn, J, D, 1)
    ts_dn = compute_ts(chi_up, chi_dn, J, D, -1)

    # Compute eigenvalues
    Enks_up, Enks_ndiag_up = diag_func(kx, ky, la, s = 1, B = B, ts = ts_up)[0:2]
    Enks_dn, Enks_ndiag_dn = diag_func(kx, ky, la, s = -1, B = B, ts = ts_dn)[0:2]

    # Compute residual_Chi
    (chi_up_new, chi_dn_new) = compute_chi(Enks_up, Enks_dn, Enks_ndiag_up, Enks_ndiag_dn, ts_up, ts_dn, T)
    # residual_chi_up = chi_up - chi_up_new
    residual_chi_dn = chi_dn - chi_dn_new

    return (residual_chi_dn)

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


la = 2.5

## diff_chi_up vs chi_up @ chi_dn fixed :::::::::::::::::::::::::::::::::::::#

fig , axes = plt.subplots(1,1, figsize=(9.2, 5.6)) # figsize is w x h in inch of figure
fig.subplots_adjust(left = 0.17, right = 0.81, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

for tick in axes.xaxis.get_major_ticks():
    tick.set_pad(7)
for tick in axes.yaxis.get_major_ticks():
    tick.set_pad(8)



#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

chi_dn_array = np.arange(-0.5, 0, 0.1)

chi_up_array = np.arange(-0.5, 0, 0.01)


diff_chi_up_array = np.zeros((len(chi_up_array)), dtype = float)
chi_up_root_list = []


cmap = mpl.cm.get_cmap("jet", len(chi_dn_array))
colors = cmap(np.arange(len(chi_dn_array)))
fig.text(0.83,0.92, r"$\lambda$ = " + str(la), fontsize = 16)
fig.text(0.83,0.87, r"$\chi_{\rm \downarrow}$ = ", fontsize = 16)
axes.axhline(y=0, ls ="--", c ="k", linewidth=0.6)


n_max = (len(chi_up_array)) * (len(chi_dn_array))
l = 0


for k, chi_dn in enumerate(chi_dn_array):
    for i, chi_up in enumerate(chi_up_array):

        ts_up = compute_ts(chi_up, chi_dn, J, D, 1)
        ts_dn = compute_ts(chi_up, chi_dn, J, D, -1)

        Enks_up, Enks_ndiag_up = diag_func(kx, ky, la, s = 1, B = B, ts = ts_up)[0:2]
        Enks_dn, Enks_ndiag_dn = diag_func(kx, ky, la, s = -1, B = B, ts = ts_dn)[0:2]

        (chi_up_new, chi_dn_new) = compute_chi(Enks_up, Enks_dn, Enks_ndiag_up, Enks_ndiag_dn, ts_up, ts_dn, T)

        diff_chi_up_array[i] = chi_up_new - chi_up

        l += 1
        print("n_iter / n_max = " + str(l) + " / " + str(n_max))

    p_residual_chi_up = partial(residual_chi_up, chi_dn = chi_dn, la = la, kx = kx, ky = ky, B = B, T = T)
    chi_up_max = optimize.fmin(p_residual_chi_up, -0.01, disp = False)
    sol_object = optimize.root(p_residual_chi_up, chi_up_max)
    chi_up_root_list.append(float(sol_object.x))


    #///// Plot /////#
    line = axes.plot(chi_up_array, diff_chi_up_array)
    plt.setp(line, ls = "-", c = colors[k], lw = 3, marker = "", mfc = 'w', ms = 6.5, mec = colors[k], mew = 2.5)
    fig.text(0.9,0.87-k*0.04, r"{0:g}".format(chi_dn), color =colors[k], fontsize = 16)
    line = axes.plot(chi_up_root_list[k], 0)
    plt.setp(line, ls = "", c = 'k', lw = 3, marker = "o", mfc = colors[k], ms = 6.5, mec = 'k', mew = 2.5)

axes.set_ylim(-0.4,0.4) # leave the ymax auto, but fix ymin
axes.locator_params(axis = 'x', nbins = 6)
axes.locator_params(axis = 'y', nbins = 6)
axes.set_xlabel(r"$\chi_{\rm \uparrow}$", labelpad = 8)
axes.set_ylabel(r"$\Delta\chi_{\rm \uparrow}$", labelpad = 8)


## Find roots for 2D function of chi using maximum of chi_up & chi_dn as root_guess

p_residual_chi_up = partial(residual_chi_up, chi_dn = -0.4, la = la, kx = kx, ky = ky, B = B, T = T)
chi_up_max = optimize.fmin(p_residual_chi_up, -0.01, disp = False)
print("max chi_up" + str(chi_up_max))

p_residual_chi_dn = partial(residual_chi_dn, chi_up = -0.4, la = la, kx = kx, ky = ky, B = B, T = T)
chi_dn_max = optimize.fmin(p_residual_chi_dn, -0.01, disp = False)
print("max chi_dn" + str(chi_dn_max))

p_residual_chi = partial(residual_chi, la = la, kx = kx, ky = ky, B = B, T = T)
sol_object = optimize.root(p_residual_chi, np.array([chi_up_max, chi_dn_max]))
chi_roots = sol_object.x
print("roots " + str(chi_roots))










plt.show()



