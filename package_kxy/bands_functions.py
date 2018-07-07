# -*- coding: Latin-1 -*-

## Modules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
import numpy as np
from numpy import cos, sin, pi, sqrt, exp
from numpy.linalg import multi_dot
np.set_printoptions(6,suppress=True,sign="+",floatmode="fixed")
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

## Universal Constant :::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
# hbar = 1.05457173e-34 # J.s
# e = 1.60217657e-19 # coulombs
# kB = 1.380648e-23 # J / K
kB = 1
## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

## Full Hamiltonian
def hamiltonian(kx, ky, la, s, B, ts):

    k1 = 1/2 * (-1 * kx - sqrt(3) * ky)
    k2 = kx
    k3 = 1/2 * (-1 * kx + sqrt(3) * ky)

    c1 = cos(k1)
    c2 = cos(k2)
    c3 = cos(k3)

    tsc = np.conj(ts)

    len_kx = np.shape(kx)[0]
    len_ky = np.shape(ky)[1]

    diagonal = (la - s * B) * np.ones((len_kx, len_ky))

    Hks = np.array([[diagonal, ts * c1 , tsc * c3],
                    [tsc * c1, diagonal, ts * c2 ],
                    [ts * c3 , tsc * c2, diagonal]]) # shape (3, 3, len_kx, len_ky)

    ## Move axis
    Hks = np.moveaxis(Hks, [-2, -1], [0, 1]) # shape (len_kx, len_ky, 3, 3)

    return Hks

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

## Derivative dH/dkx
def derivative_kx_hamiltonian(kx, ky, la, s, B, ts):

    k1 = 1/2 * (-1 * kx - sqrt(3) * ky)
    k2 = kx
    k3 = 1/2 * (-1 * kx + sqrt(3) * ky)

    s1 = 0.5 * sin(k1)
    s2 = - sin(k2)
    s3 = 0.5 * sin(k3)

    tsc = np.conj(ts)

    len_kx = np.shape(kx)[0]
    len_ky = np.shape(ky)[1]

    diagonal = np.zeros((len_kx, len_ky))

    dHks_dkx = np.array([[diagonal, ts * s1 , tsc * s3],
                        [tsc * s1, diagonal, ts * s2 ],
                        [ts * s3 , tsc * s2, diagonal]]) # shape (3, 3, len_kx, len_ky)

    ## Move axis
    dHks_dkx = np.moveaxis(dHks_dkx, [-2, -1], [0, 1]) # shape (len_kx, len_ky, 3, 3)

    return dHks_dkx

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

## Derivative dH/dky
def derivative_ky_hamiltonian(kx, ky, la, s, B, ts):

    k1 = 1/2 * (-1 * kx - sqrt(3) * ky)
    k2 = kx
    k3 = 1/2 * (-1 * kx + sqrt(3) * ky)

    s1 = sqrt(3) / 2 * sin(k1)
    s2 = 0 * sin(k2)
    s3 = - sqrt(3) / 2 * sin(k3)

    tsc = np.conj(ts)

    len_kx = np.shape(kx)[0]
    len_ky = np.shape(ky)[1]

    diagonal = np.zeros((len_kx, len_ky))

    dHks_dky = np.array([[diagonal, ts * s1 , tsc * s3],
                        [tsc * s1, diagonal, ts * s2 ],
                        [ts * s3 , tsc * s2, diagonal]]) # shape (3, 3, len_kx, len_ky)

    ## Move axis
    dHks_dky = np.moveaxis(dHks_dky, [-2, -1], [0, 1]) # shape (len_kx, len_ky, 3, 3)

    return dHks_dky

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

## Diaganolization function
def diag_func(kx, ky, la, s, B, ts):

    """
    (i,j) -> (kx, ky)
    n -> different eigenvalues E[n]
    l -> different components of eigenvectors V[:, n]
    """

    Hks = hamiltonian(kx, ky, la, s, B, ts)
    dHks_dkx = derivative_kx_hamiltonian(kx, ky, la, s, B, ts)
    dHks_dky = derivative_ky_hamiltonian(kx, ky, la, s, B, ts)

    Enks, Vnks = np.linalg.eigh(Hks)
    # The column V[:, n] is the normalized eigenvector corresponding to the eigenvalue E[n]

    # Eigen values of non-diagonal part of the hamiltonian
    Enks_ndiag = Enks - ( la - s * B )

    # Compute la_min for all Enks > 0
    la_min = s * B - np.min(Enks_ndiag)

    return Enks, Enks_ndiag, Vnks, dHks_dkx, dHks_dky, la_min

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

## Hopping term
def compute_ts(chi_up, chi_dn, J, D, s):

    if s == 1:
        return J * ( chi_up + chi_dn ) - 1j * s * D * chi_dn
    else:
        return J * ( chi_up + chi_dn ) - 1j * s * D * chi_up

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

    S = (sum_s_up + sum_s_dn) / Nt / 2

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

## Fit function
def residual_S_chi(pars, kx, ky, B, J, D, T):


    la = pars[0]
    chi_up = pars[1]
    chi_dn = pars[2]

    print("[lambda, chi_up, chi_dn] =     ", np.array([la, chi_up, chi_dn]))

    # Compute ts_up & ts_dn
    ts_up = compute_ts(chi_up, chi_dn, J, D, 1)
    ts_dn = compute_ts(chi_up, chi_dn, J, D, -1)

    # Compute eigenvalues
    Enks_up_new, Enks_ndiag_up_new = diag_func(kx, ky, la, s = 1, B = B, ts = ts_up)[0:2]
    Enks_dn_new, Enks_ndiag_dn_new = diag_func(kx, ky, la, s = -1, B = B, ts = ts_dn)[0:2]

    # Compute residual_Chi
    (chi_up_new, chi_dn_new) = compute_chi(Enks_up_new, Enks_dn_new, Enks_ndiag_up_new, Enks_ndiag_dn_new, ts_up, ts_dn, T)
    residual_chi_up = (chi_up - chi_up_new)
    residual_chi_dn = (chi_dn - chi_dn_new)

    # Compute residual_S
    residual_S = compute_S(Enks_up_new, Enks_dn_new, T) - 0.5

    print("residual [S, chi_up, chi_dn] = ", np.array([residual_S, residual_chi_up, residual_chi_dn]))


    return (residual_S, residual_chi_up, residual_chi_dn)


## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

## Berry phase

def berry_phase(Enks, Vnks, dHks_dkx, dHks_dky):

    """
    (i,j) -> (kx, ky)
    n -> different eigenvalues E[n]
    l -> different components of eigenvectors V[:, n]
    """

    len_kx = np.shape(Enks)[0]
    len_ky = np.shape(Enks)[1]

    dVnks_dkx = np.zeros((len_kx, len_ky, 3, 3), dtype = complex) # # dim: i, j, l, n
    dVnks_dky = np.zeros((len_kx, len_ky, 3, 3), dtype = complex) # # dim: i, j, l, n

    Omega_nks = np.zeros((len_kx, len_ky, 3), dtype = float) # dim: i, j, n

    for i in range(len_kx):
        for j in range(len_ky):

                dVnks_dkx[i,j,:,0] = multi_dot([np.conj(Vnks[i,j,:,1]), dHks_dkx[i,j,:,:], Vnks[i,j,:,0]]) / (Enks[i,j,0] - Enks[i,j,1]) * Vnks[i,j,:,1]  \
                                   + multi_dot([np.conj(Vnks[i,j,:,2]), dHks_dkx[i,j,:,:], Vnks[i,j,:,0]]) / (Enks[i,j,0] - Enks[i,j,2]) * Vnks[i,j,:,2]
                dVnks_dkx[i,j,:,1] = multi_dot([np.conj(Vnks[i,j,:,0]), dHks_dkx[i,j,:,:], Vnks[i,j,:,1]]) / (Enks[i,j,1] - Enks[i,j,0]) * Vnks[i,j,:,0]  \
                                   + multi_dot([np.conj(Vnks[i,j,:,2]), dHks_dkx[i,j,:,:], Vnks[i,j,:,1]]) / (Enks[i,j,1] - Enks[i,j,2]) * Vnks[i,j,:,2]
                dVnks_dkx[i,j,:,2] = multi_dot([np.conj(Vnks[i,j,:,0]), dHks_dkx[i,j,:,:], Vnks[i,j,:,2]]) / (Enks[i,j,2] - Enks[i,j,0]) * Vnks[i,j,:,0]  \
                                   + multi_dot([np.conj(Vnks[i,j,:,1]), dHks_dkx[i,j,:,:], Vnks[i,j,:,2]]) / (Enks[i,j,2] - Enks[i,j,1]) * Vnks[i,j,:,1]

                dVnks_dky[i,j,:,0] = multi_dot([np.conj(Vnks[i,j,:,1]), dHks_dky[i,j,:,:], Vnks[i,j,:,0]]) / (Enks[i,j,0] - Enks[i,j,1]) * Vnks[i,j,:,1]  \
                                   + multi_dot([np.conj(Vnks[i,j,:,2]), dHks_dky[i,j,:,:], Vnks[i,j,:,0]]) / (Enks[i,j,0] - Enks[i,j,2]) * Vnks[i,j,:,2]
                dVnks_dky[i,j,:,1] = multi_dot([np.conj(Vnks[i,j,:,0]), dHks_dky[i,j,:,:], Vnks[i,j,:,1]]) / (Enks[i,j,1] - Enks[i,j,0]) * Vnks[i,j,:,0]  \
                                   + multi_dot([np.conj(Vnks[i,j,:,2]), dHks_dky[i,j,:,:], Vnks[i,j,:,1]]) / (Enks[i,j,1] - Enks[i,j,2]) * Vnks[i,j,:,2]
                dVnks_dky[i,j,:,2] = multi_dot([np.conj(Vnks[i,j,:,0]), dHks_dky[i,j,:,:], Vnks[i,j,:,2]]) / (Enks[i,j,2] - Enks[i,j,0]) * Vnks[i,j,:,0]  \
                                   + multi_dot([np.conj(Vnks[i,j,:,1]), dHks_dky[i,j,:,:], Vnks[i,j,:,2]]) / (Enks[i,j,2] - Enks[i,j,1]) * Vnks[i,j,:,1]

                Omega_nks[i,j,0] = 2 * np.real( 1j * np.dot(np.conj(dVnks_dkx[i,j,:,0]),dVnks_dky[i,j,:,0]))
                Omega_nks[i,j,1] = 2 * np.real( 1j * np.dot(np.conj(dVnks_dkx[i,j,:,1]),dVnks_dky[i,j,:,1]))
                Omega_nks[i,j,2] = 2 * np.real( 1j * np.dot(np.conj(dVnks_dkx[i,j,:,2]),dVnks_dky[i,j,:,2]))

    return Omega_nks

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#