# This file is part of pyMathHelpers.
# pyMathHelpers is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
# version.
# pyMathHelpers is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with pyMathHelpers. If not, see
# <https://www.gnu.org/licenses/>.

import numpy as np
from math import sqrt, pi

# we try to use numba to speed things up considerably
USE_NUMBA = True

try:
    if USE_NUMBA:
        from numba import njit
    else:
        raise ImportError
except ImportError:
    def njit(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func


# faster internal version of it
@njit
def lm2idx(l, m):
    return (l * (l + 1)) // 2 + m


@njit
def outSize(maxl):
    return (maxl + 1) * (maxl + 2) // 2


@njit
def idx2lm(idx):
    """
    returns the l an m given a position in an output array
    :param idx: index in flat array
    :return: index l, in
    """
    l = (-1 + sqrt(1 + 8 * idx)) // 2
    m = idx - (l * (1 + l)) // 2
    return l, m


# this is only needed for normal spherical harmonics
@njit
def computeAlpha(lmax):
    alpha = [[0.0 for j in range(0, lmax+1)] for i in range(0, lmax+1)]
    
    fourpi = 4 * pi;
    
    for l in range(0, lmax+1):
        alpha[l, 0] = sqrt((2*l+1)/fourpi)
        for m in range(1, l+1):
              alpha[l, m] = -alpha[l, m-1]/sqrt((l+m)*(l+1-m))

    return alpha


def getCosSinMap(lambd, lmax):
    m = np.arange(0, lmax + 1)
    angles = m * lambd
    cosmap = np.cos(angles)
    sinmap = np.sin(angles)
    return cosmap, sinmap


@njit
def getBetas(lmax):
    fpi = 4 * np.pi
    size = (lmax + 1) * (lmax + 2) // 2
    betas = [0.0 for i in range(0, size)]
    
    for l in range(1, lmax + 1):
        lm2idx_l0 = lm2idx(l, 0)
        betas[lm2idx_l0] = sqrt((2 * l + 1) / fpi / (l * (l + 1)))
        for m in range(1, l + 1):
            lm2idx_lm = lm2idx(l, m)
            lm2idx_lm1 = lm2idx(l, m - 1)
            betas[lm2idx_lm] = -betas[lm2idx_lm1] / sqrt((l + m) * (l + 1 - m))

    return betas

@njit
def getAB(x, lmax):
    sx = sqrt(1 - x * x)
    size = (lmax + 1) * (lmax + 2) // 2
    A = [0.0 for i in range(0, size)]
    B = [0.0 for i in range(0, size)]

    lm2idx_11 = lm2idx(1, 1)
    B[lm2idx_11] = 1.0
    for m in range(1, lmax):
        lm2idx_m1m1 = lm2idx(m + 1, m + 1)
        lm2idx_mm = lm2idx(m, m)
        lm2idx_m1m = lm2idx(m + 1, m)
        
        B[lm2idx_m1m1] = ((2. * m + 1.) * (m + 1.) / m) * sx * B[lm2idx_mm]
        B[lm2idx_m1m] = (2. * m + 1.) * x * B[lm2idx_mm]

    for m in range(1, lmax - 1):
        for l in range(m + 2, lmax + 1):
            lm2idx_lm = lm2idx(l, m)
            lm2idx_l1m = lm2idx(l - 1, m)
            lm2idx_l2m = lm2idx(l - 2, m)
            B[lm2idx_lm] = ((2. * l - 1.) * x * B[lm2idx_l1m] - (l - 1 + m) * B[lm2idx_l2m]) / (l - m)

    for l in range(1, lmax + 1):
        lm2idx_l0 = lm2idx(l, 0)
        lm2idx_l1 = lm2idx(l, 1)
        A[lm2idx_l0] = sx * B[lm2idx_l1]

    for l in range(1, lmax + 1):
        for m in range(1, l + 1):
            lm2idx_lm = lm2idx(l, m)
            if l == m:
                A[lm2idx_lm] = -l * x * B[lm2idx_lm] / m
            else:
                lm2idx_l1m = lm2idx(l - 1, m)
                A[lm2idx_lm] = (-l * x * B[lm2idx_lm] + (l + m) * B[lm2idx_l1m]) / m

    return [A, B]


def computeAssociatedLegendre(x, lmax):
    sx = sqrt(1 - x**2)
    P = np.zeros((lmax+1, lmax+1))
    P[0, 0] = 1.0

    # Compute diagonal terms P[m+1, m+1]
    m_vals = np.arange(0, lmax)
    P[m_vals+1, m_vals+1] = (2*m_vals + 1) * sx * np.diag(P)[m_vals]

    # Compute edge terms P[m+1, m]
    P[m_vals+1, m_vals] = (2*m_vals + 1) * x * np.diag(P)[m_vals]

    # Compute remaining terms using recurrence
    for m in range(0, lmax - 1):
        l_vals = np.arange(m + 2, lmax + 1)
        l = l_vals[:, np.newaxis]
        P[l_vals, m] = ((2*l_vals - 1)*x*P[l_vals - 1, m] - (l_vals - 1 + m)*P[l_vals - 2, m]) / (l_vals - m)

    return P


@njit
def computeF(alpha, delta, lmax):
    alp = computeAlpha(lmax)
    cosSin = getCosSinMap(alpha, lmax)
    P = computeAssociatedLegendre(np.sin(delta), lmax)
    F = np.zeros((lmax + 1)**2)
    i = 0
    
    for l in range(0, lmax+1):
        F[i] = alp[l,0]*P[l,0]
        i = i+1
        for m in range(1, l+1):
            F[i] = 2 * alp[l,m] * P[l,m] * cosSin[0][m]
            i = i+1
            F[i] = -2 * alp[l,m] * P[l,m] * cosSin[1][m]
            i = i+1
            
    return F
