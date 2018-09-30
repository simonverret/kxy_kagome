# -*- coding: Latin-1 -*-

## Modules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
import numpy as np
from numpy import cos, sin, pi, sqrt, exp, log, conj, dot
from scipy.special import spence as dilog
np.set_printoptions(6,suppress=True,sign="+",floatmode="fixed")
from functools import partial
from scipy import optimize
from numba import jit
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

## Universal Constant :::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
# hbar = 1.05457173e-34 # J.s
# e = 1.60217657e-19 # coulombs
# kB = 1.380648e-23 # J / K
hbar = 1
kB = 1
## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

## Hamiltonian
def hamiltonian(kx, ky, la, s, B, ts):

    k1 = 1/2 * (-1 * kx - sqrt(3) * ky)
    k2 = kx
    k3 = 1/2 * (-1 * kx + sqrt(3) * ky)

    c1 = cos(k1)
    c2 = cos(k2)
    c3 = cos(k3)

    tsc = conj(ts)

    len_kx = kx.shape[0]
    len_ky = ky.shape[1]

    diagonal = (la - s * B) * np.ones((len_kx, len_ky))

    Hks = np.array([[diagonal, ts * c1 , tsc * c3],
                    [tsc * c1, diagonal, ts * c2 ],
                    [ts * c3 , tsc * c2, diagonal]]) # shape (3, 3, len_kx, len_ky)

    ## Move axis for diagonalization
    Hks = np.moveaxis(Hks, [-2, -1], [0, 1]) # redistribute axis positions with shape (len_kx, len_ky, 3, 3)

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

    tsc = conj(ts)

    len_kx = kx.shape[0]
    len_ky = ky.shape[1]

    diagonal = np.zeros((len_kx, len_ky))

    dHks_dkx = np.array([[diagonal, ts * s1 , tsc * s3],
                        [tsc * s1, diagonal, ts * s2 ],
                        [ts * s3 , tsc * s2, diagonal]]) # shape (3, 3, len_kx, len_ky)

    ## Move axis
    dHks_dkx = np.moveaxis(dHks_dkx, [-2, -1], [0, 1]) # redistribute axis positions with shape (len_kx, len_ky, 3, 3)

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

    tsc = conj(ts)

    len_kx = kx.shape[0]
    len_ky = ky.shape[1]

    diagonal = np.zeros((len_kx, len_ky))

    dHks_dky = np.array([[diagonal, ts * s1 , tsc * s3],
                        [tsc * s1, diagonal, ts * s2 ],
                        [ts * s3 , tsc * s2, diagonal]]) # shape (3, 3, len_kx, len_ky)

    ## Move axis
    dHks_dky = np.moveaxis(dHks_dky, [-2, -1], [0, 1]) # redistribute axis positions with shape (len_kx, len_ky, 3, 3)

    return dHks_dky

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

## Diaganolization function
def diag_func(kx, ky, la, s, B, ts):

    """
    Input:
    kx and kxy must be meshgrid matrix

    Returns:
    Enks.shape = (i, j, n)
    Enks_non_diag.shape = (i, j, n)
    Vnks_diag.shape = (i, j, l, n)

    (i,j) -> (kx, ky)
    n -> different eigenvalues E[n]
    l -> different components of eigenvectors V[:, n]
    """

    Hks = hamiltonian(kx, ky, la, s, B, ts)
    dHks_dkx = derivative_kx_hamiltonian(kx, ky, la, s, B, ts)
    dHks_dky = derivative_ky_hamiltonian(kx, ky, la, s, B, ts)

    Enks, Vnks = np.linalg.eigh(Hks) # does always diagonalizaton on the two last axis
    # The column V[:, n] is the normalized eigenvector corresponding to the eigenvalue E[n]

    # Eigen values of non-diagonal part of the hamiltonian
    Enks_ndiag = Enks - ( la - s * B )

    # Compute la_min for all Enks > 0
    la_min = s * B - np.min(Enks_ndiag)

    return Enks, Enks_ndiag, Vnks, dHks_dkx, dHks_dky, la_min

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

## Diaganolization function
def diag_for_chi_func(kx, ky):

    """
    Returns:
    fnk.shape = (i, j, n)

    (i,j) -> (kx, ky)
    n -> different eigenvalues E[n]
    l -> different components of eigenvectors V[:, n]
    """

    len_kx = kx.shape[0]
    len_ky = ky.shape[1]

    Hks_ndiag_t_1 = hamiltonian(kx, ky, la = 0, s = 0, B = 0, ts = 1)
    Hks_ndiag_t_j = hamiltonian(kx, ky, la = 0, s = 0, B = 0, ts = 1j)

    fnk_t_1, Vnks_t_1 = np.linalg.eigh(Hks_ndiag_t_1)
    fnk_t_j, Vnks_t_j = np.linalg.eigh(Hks_ndiag_t_j)

    fnk_tot = np.empty((len_kx, len_ky, 3))
    fnk_tot[:, :, 0] = fnk_t_1[:, :, 0] + fnk_t_j[:, :, 0]
    fnk_tot[:, :, 1] = fnk_t_1[:, :, 1] + fnk_t_j[:, :, 1]
    fnk_tot[:, :, 2] = fnk_t_1[:, :, 2] + fnk_t_j[:, :, 2]

    return fnk_tot

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

    # index_pos = x > 0
    # index_neg = x <= 0

    # nB = np.zeros(x.shape)

    # nB[index_pos] = 1 / (exp(x[index_pos]) - 1)
    # nB[index_neg] = 0

    nB = 1 / (exp(x) - 1)

    return nB

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

## Compute S = sum(Enks/kB*T)
def compute_S(Enks_up, Enks_dn, T):

    Nt = Enks_up.shape[0] * Enks_up.shape[1]

    sum_s_up = np.sum(n_B(Enks_up[:,:,0] / (kB * T))) + np.sum(n_B(Enks_up[:,:,1] / (kB * T))) + np.sum(n_B(Enks_up[:,:,2] / (kB * T)))
    sum_s_dn = np.sum(n_B(Enks_dn[:,:,0] / (kB * T))) + np.sum(n_B(Enks_dn[:,:,1] / (kB * T))) + np.sum(n_B(Enks_dn[:,:,2] / (kB * T)))

    S = (sum_s_up + sum_s_dn) / Nt / 6

    return S

def compute_chi(Enks_up, Enks_dn, fnk_tot, ts_up, ts_dn, T):

    Nt = Enks_up.shape[0] * Enks_up.shape[1]

    sum_s_up = np.sum(fnk_tot[:,:,0] * n_B(Enks_up[:,:,0] / (kB * T))) \
             + np.sum(fnk_tot[:,:,1] * n_B(Enks_up[:,:,1] / (kB * T))) \
             + np.sum(fnk_tot[:,:,2] * n_B(Enks_up[:,:,2] / (kB * T)))

    sum_s_dn = np.sum(fnk_tot[:,:,0] * n_B(Enks_dn[:,:,0] / (kB * T))) \
             + np.sum(fnk_tot[:,:,1] * n_B(Enks_dn[:,:,1] / (kB * T))) \
             + np.sum(fnk_tot[:,:,2] * n_B(Enks_dn[:,:,2] / (kB * T)))

    chi_up = sum_s_up / Nt / 6
    chi_dn = sum_s_dn / Nt / 6

    return (chi_up, chi_dn)


## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

## Fit function
def residual_S_chi(pars, kx, ky, B, J, D, T):

    la = pars[0]
    chi_up = pars[1]
    chi_dn = pars[2]

    # print("[lambda, chi_up, chi_dn] =     ", np.array([la, chi_up, chi_dn]))

    # Compute ts_up & ts_dn
    ts_up = compute_ts(chi_up, chi_dn, J, D, 1)
    ts_dn = compute_ts(chi_up, chi_dn, J, D, -1)

    # Compute eigenvalues
    Enks_up_new = diag_func(kx, ky, la, s = 1, B = B, ts = ts_up)[0]
    Enks_dn_new = diag_func(kx, ky, la, s = -1, B = B, ts = ts_dn)[0]

    # Compute fnk_tot
    fnk_tot = diag_for_chi_func(kx, ky)

    # Compute residual_Chi
    (chi_up_new, chi_dn_new) = compute_chi(Enks_up_new, Enks_dn_new, fnk_tot, ts_up, ts_dn, T)
    residual_chi_up = (chi_up - chi_up_new)
    residual_chi_dn = (chi_dn - chi_dn_new)

    # Compute residual_S
    residual_S = compute_S(Enks_up_new, Enks_dn_new, T) - 0.5

    # print("residual [S, chi_up, chi_dn] = ", np.array([residual_S, residual_chi_up, residual_chi_dn]))


    return (residual_S, residual_chi_up, residual_chi_dn)


## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

## Berry phase
@jit(nopython=True)
def berry_phase(Enks, Vnks, dHks_dkx, dHks_dky):

    """
    Returns:
    Omega_nks.shape = (i, j, n)

    (i,j) -> (kx, ky)
    n -> different eigenvalues E[n]
    """

    len_kx = Enks.shape[0]
    len_ky = Enks.shape[1]

    dVnks_dkx = np.zeros((len_kx, len_ky, 3, 3), dtype = np.complex64) # # dim: i, j, l, n
    dVnks_dky = np.zeros((len_kx, len_ky, 3, 3), dtype = np.complex64) # # dim: i, j, l, n

    Omega_nks = np.zeros((len_kx, len_ky, 3), dtype = np.float64) # dim: i, j, n

    for i in range(len_kx):
        for j in range(len_ky):

                dVnks_dkx[i,j,:,0] = conj(Vnks[i,j,:,1]).dot(dHks_dkx[i,j,:,:]).dot(Vnks[i,j,:,0]) / (Enks[i,j,0] - Enks[i,j,1]) * Vnks[i,j,:,1]  \
                                   + conj(Vnks[i,j,:,2]).dot(dHks_dkx[i,j,:,:]).dot(Vnks[i,j,:,0]) / (Enks[i,j,0] - Enks[i,j,2]) * Vnks[i,j,:,2]
                dVnks_dkx[i,j,:,1] = conj(Vnks[i,j,:,0]).dot(dHks_dkx[i,j,:,:]).dot(Vnks[i,j,:,1]) / (Enks[i,j,1] - Enks[i,j,0]) * Vnks[i,j,:,0]  \
                                   + conj(Vnks[i,j,:,2]).dot(dHks_dkx[i,j,:,:]).dot(Vnks[i,j,:,1]) / (Enks[i,j,1] - Enks[i,j,2]) * Vnks[i,j,:,2]
                dVnks_dkx[i,j,:,2] = conj(Vnks[i,j,:,0]).dot(dHks_dkx[i,j,:,:]).dot(Vnks[i,j,:,2]) / (Enks[i,j,2] - Enks[i,j,0]) * Vnks[i,j,:,0]  \
                                   + conj(Vnks[i,j,:,1]).dot(dHks_dkx[i,j,:,:]).dot(Vnks[i,j,:,2]) / (Enks[i,j,2] - Enks[i,j,1]) * Vnks[i,j,:,1]

                dVnks_dky[i,j,:,0] = conj(Vnks[i,j,:,1]).dot(dHks_dky[i,j,:,:]).dot(Vnks[i,j,:,0]) / (Enks[i,j,0] - Enks[i,j,1]) * Vnks[i,j,:,1]  \
                                   + conj(Vnks[i,j,:,2]).dot(dHks_dky[i,j,:,:]).dot(Vnks[i,j,:,0]) / (Enks[i,j,0] - Enks[i,j,2]) * Vnks[i,j,:,2]
                dVnks_dky[i,j,:,1] = conj(Vnks[i,j,:,0]).dot(dHks_dky[i,j,:,:]).dot(Vnks[i,j,:,1]) / (Enks[i,j,1] - Enks[i,j,0]) * Vnks[i,j,:,0]  \
                                   + conj(Vnks[i,j,:,2]).dot(dHks_dky[i,j,:,:]).dot(Vnks[i,j,:,1]) / (Enks[i,j,1] - Enks[i,j,2]) * Vnks[i,j,:,2]
                dVnks_dky[i,j,:,2] = conj(Vnks[i,j,:,0]).dot(dHks_dky[i,j,:,:]).dot(Vnks[i,j,:,2]) / (Enks[i,j,2] - Enks[i,j,0]) * Vnks[i,j,:,0]  \
                                   + conj(Vnks[i,j,:,1]).dot(dHks_dky[i,j,:,:]).dot(Vnks[i,j,:,2]) / (Enks[i,j,2] - Enks[i,j,1]) * Vnks[i,j,:,1]

                Omega_nks[i,j,0] = 2 * np.real( 1j * np.dot(conj(dVnks_dkx[i,j,:,0]),dVnks_dky[i,j,:,0]))
                Omega_nks[i,j,1] = 2 * np.real( 1j * np.dot(conj(dVnks_dkx[i,j,:,1]),dVnks_dky[i,j,:,1]))
                Omega_nks[i,j,2] = 2 * np.real( 1j * np.dot(conj(dVnks_dkx[i,j,:,2]),dVnks_dky[i,j,:,2]))

    return Omega_nks


## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
def c2_func(x):
    c2 = ( 1 + x ) * ( log ( (1 + x) / x ) )**2 - ( log(x) )**2 - 2 * dilog(1-(-x))
    # dilog from scipy.special.spence is defined with a different convention where z = 1 - x
    return c2

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

## Kxy function
def kxy_single_value(Enks_up, Enks_dn, Omega_nks_up, Omega_nks_dn, T):

    """
    Returns:
    Omega_nks.shape = (i, j, n)

    (i,j) -> (kx, ky)
    n -> different eigenvalues E[n]
    """

    len_kx = Enks_up.shape[0]
    len_ky = Enks_up.shape[1]

    Nt = len_kx * len_ky

    coeff_up = np.empty((len_kx, len_ky, 3), dtype = np.float64) # dim: i, j, n
    coeff_dn = np.empty((len_kx, len_ky, 3), dtype = np.float64) # dim: i, j, n

    coeff_up[:, :, 0] = ( c2_func( n_B( Enks_up[:, :, 0] / (kB * T) ) ) - pi**2 / 3 ) * Omega_nks_up[:, :, 0]
    coeff_up[:, :, 1] = ( c2_func( n_B( Enks_up[:, :, 1] / (kB * T) ) ) - pi**2 / 3 ) * Omega_nks_up[:, :, 1]
    coeff_up[:, :, 2] = ( c2_func( n_B( Enks_up[:, :, 2] / (kB * T) ) ) - pi**2 / 3 ) * Omega_nks_up[:, :, 2]

    coeff_dn[:, :, 0] = ( c2_func( n_B( Enks_dn[:, :, 0] / (kB * T) ) ) - pi**2 / 3 ) * Omega_nks_dn[:, :, 0]
    coeff_dn[:, :, 1] = ( c2_func( n_B( Enks_dn[:, :, 1] / (kB * T) ) ) - pi**2 / 3 ) * Omega_nks_dn[:, :, 1]
    coeff_dn[:, :, 2] = ( c2_func( n_B( Enks_dn[:, :, 2] / (kB * T) ) ) - pi**2 / 3 ) * Omega_nks_dn[:, :, 2]

    kxy = - ( kB**2 * T ) / (hbar * Nt ) * \
            ( np.sum(coeff_up[:,:,0] + coeff_up[:,:,1] + coeff_up[:,:,2]) \
            + np.sum(coeff_dn[:,:,0] + coeff_dn[:,:,1] + coeff_dn[:,:,2]) )

    return kxy


def kxy_algorithm(kxx, kyy, B, D, J, T, la_min, chi_up_ini, chi_dn_ini, steps_on_chi_ini = False):

    p_residual_S_chi = partial(residual_S_chi, kx = kxx, ky = kyy, B = B, D = D, J = J, T = T)

    if steps_on_chi_ini == True:
        # In order to avoid the trivial value for (chi_up, chi_dn) = (0,0), we look for
        # chi roots different from the trivial ones by trying different chi_ini values
        # starting from chi_ini = 0 to higher values, as the non trivial roots are the second
        # roots to find before chi_function becomes discontinous:

        chi_steps = np.arange(-5, 0, 0.1)[::-1]
        for i, chi_ini in enumerate(chi_steps):
                print("Attent " + str(i) + ": chi_ini = " + str(chi_ini))
                out = optimize.root(p_residual_S_chi, np.array([la_min, chi_ini, chi_ini]))
                roots = out.x

                if np.all(np.abs(roots[1:]) < 1e-4) or (out.success is False) : # (chi_up, chi_dn) < 1e-4
                    continue
                else:
                    break

    else:
        out = optimize.root(p_residual_S_chi, np.array([la_min, chi_up_ini, chi_dn_ini]))
        roots = out.x

    la = roots[0]
    chi_up = roots[1]
    chi_dn = roots[2]

    ## Compute bands from the right lambda, chi_up and chi_dn
    ts_up = compute_ts(chi_up, chi_dn, J, D, 1)
    ts_dn = compute_ts(chi_up, chi_dn, J, D, -1)
    Enks_up, Enks_ndiag_up, Vnks_up, dHks_dkx_up, dHks_dky_up = diag_func(kxx, kyy, la, s = 1, B = B, ts = ts_up)[0:-1]
    Enks_dn, Enks_ndiag_dn, Vnks_dn, dHks_dkx_dn, dHks_dky_dn = diag_func(kxx, kyy, la, s = -1, B = B, ts = ts_dn)[0:-1]
    Omega_nks_up = berry_phase(Enks_up, Vnks_up, dHks_dkx_up, dHks_dky_up)
    Omega_nks_dn = berry_phase(Enks_dn, Vnks_dn, dHks_dkx_dn, dHks_dky_dn)

    ## Compute kxy from bands and Berry phase
    kxy = kxy_single_value(Enks_up, Enks_dn, Omega_nks_up, Omega_nks_dn, T)

    ## Display results
    print("[T, la, chi_up, chi_dn, kxy] = ", np.array([T, la, chi_up, chi_dn, kxy]))

    return kxy, la, chi_up, chi_dn, ts_up, ts_dn