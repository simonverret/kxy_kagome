# -*- coding: Latin-1 -*-

## Modules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
import numpy as np
from numpy import cos, sin, pi, sqrt, exp
np.set_printoptions(6,suppress=True,sign="+",floatmode="fixed")
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
from functools import partial
from scipy import integrate, optimize
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

T = 0.1

J = 1
D = 0.2 * J

la_ini = +2.215964
chi_up_ini = -0.864148
chi_dn_ini = -0.611929

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

# Contains the hexago volume in a rectangle of side pi x (2*pi/sqrt(3))
kx = np.linspace(-pi / 3, 2 * pi / 3, 10)
ky = np.linspace(-pi / sqrt(3), pi / sqrt(3), 10)

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#


#::: Initialize ::::::#

# Compute ts_up & ts_dn
ts_up_ini = bands_func.compute_ts(chi_up_ini, chi_dn_ini, J, D, 1)
ts_dn_ini = bands_func.compute_ts(chi_up_ini, chi_dn_ini, J, D, -1)

# Compute eigenvalues
Enks_up_ini, Enks_ndiag_up_ini = bands_func.diag_func(kx, ky, la_ini, s = 1, B = B, ts = ts_up_ini)[0:2]
Enks_dn_ini, Enks_ndiag_dn_ini = bands_func.diag_func(kx, ky, la_ini, s = -1, B = B, ts = ts_dn_ini)[0:2]

# Compute residual_S
S_ini = bands_func.compute_S(Enks_up_ini, Enks_dn_ini, T)

#::::::::: New ::::::#

# Compute residual_Chi
(chi_up_new, chi_dn_new) = bands_func.compute_chi(Enks_up_ini, Enks_dn_ini, Enks_ndiag_up_ini, Enks_ndiag_dn_ini, ts_up_ini, ts_dn_ini, T)

print("IN (lambda, chi_up, chi_dn) = ("
       + r"{0:.4e}".format(la_ini) + ", "
       + r"{0:.4e}".format(chi_up_ini) + ", "
       + r"{0:.4e}".format(chi_dn_ini) + ")")


print("OUT (S, chi_up, chi_dn) = ("
       + r"{0:.4e}".format(S_ini) + ", "
       + r"{0:.4e}".format(chi_up_new) + ", "
       + r"{0:.4e}".format(chi_dn_new) + ")")
