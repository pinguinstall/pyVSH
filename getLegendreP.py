# This file is part of pyMathHelpers.
# pyMathHelpers is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
# version.
# pyMathHelpers is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with pyMathHelpers. If not, see
# <https://www.gnu.org/licenses/>.

import numpy as np


def lm2idx(l: int, m: int) -> int:
    """
    returns the position in an output array regarding the l and m parameter
    :param l: order l
    :param m: grade m
    :return: index in flat array
    """
    return int((l * (1 + l)) / 2 + m)


def idx2lm(idx: int) -> (int, int):
    """
    returns the l an m given a position in an output array
    :param idx: index in flat array
    :return: index l, in
    """
    l = int(((-1 + np.sqrt(1 + 8 * idx)) / 2))
    m = int(idx - (l * (1 + l)) / 2)
    return l, m


def getOutputArraySize(maxl: int) -> int:
    """returns the size of the array needed for l_max"""
    return int((maxl + 1) * (maxl + 2) / 2)


# this is only needed for normal spherical harmonics
def computeAlpha(lmax):
    alpha = np.zeros((lmax+1, lmax+1));
    fourpi = 4*np.pi;
    for l in range(0, lmax+1):
        alpha[l, 0] = np.sqrt((2*l+1)/fourpi)
        for m in range(1, l+1):
              alpha[l, m] = -alpha[l, m-1]/np.sqrt((l+m)*(l+1-m))

    return alpha


def getCosSinMap(lambd: float, lmax: int) -> (np.array, np.array):
    cosmap = np.zeros(lmax + 1)
    sinmap = np.zeros(lmax + 1)
    cosmap[0] = 1

    c1 = np.cos(lambd)
    s1 = np.sin(lambd)
    sinmap[1] = s1
    cosmap[1] = c1

    for m in range(2, lmax + 1):
        cosmap[m] = c1 * cosmap[m - 1] - s1 * sinmap[m - 1]
        sinmap[m] = s1 * cosmap[m - 1] + c1 * sinmap[m - 1]

    return cosmap, sinmap


def getBetas(lmax: int) -> np.array:
    fpi = 4 * np.pi
    betas = np.zeros(getOutputArraySize(lmax))
    for l in range(1, lmax + 1):
        betas[lm2idx(l, 0)] = np.sqrt((2 * l + 1) / fpi / (l * (l + 1)))
        for m in range(1, l + 1):
            betas[lm2idx(l, m)] = -betas[lm2idx(l, m - 1)] / np.sqrt(
                (l + m) * (l + 1 - m))

    return betas


def getAB(x: float, lmax: int) -> (np.array, np.array):
    sx = np.sqrt(1 - x * x)
    A = np.zeros(getOutputArraySize(lmax))
    B = np.zeros(getOutputArraySize(lmax))

    #    for l in range(1, lmax + 1):
    #        B[lm2idx(l, 0)] = 0.0

    B[lm2idx(1, 1)] = 1.0
    for m in range(1, lmax):
        B[lm2idx(m + 1, m + 1)] = ((2. * m + 1.) * (m + 1.) / m) * sx * B[lm2idx(m, m)]
        B[lm2idx(m + 1, m)] = (2. * m + 1.) * x * B[lm2idx(m, m)]

    for m in range(1, lmax - 1):
        for l in range(m + 2, lmax + 1):
            B[lm2idx(l, m)] = ((2. * l - 1.) * x * B[lm2idx(l - 1, m)] - (
                    l - 1 + m) * B[lm2idx(l - 2, m)]) / (l - m)

    for l in range(1, lmax + 1):
        A[lm2idx(l, 0)] = sx * B[lm2idx(l, 1)]

    for l in range(1, lmax + 1):
        for m in range(1, l + 1):
            if l == m:
                A[lm2idx(l, m)] = -l * x * B[lm2idx(l, m)] / m
            else:
                A[lm2idx(l, m)] = (-l * x * B[lm2idx(l, m)] + (l + m) * B[lm2idx(l - 1, m)]) / m

    return A, B


def computeAssociatedLegendre(x, lmax):
    sx = np.sqrt(1 - x**2)
    P = np.zeros((lmax+1, lmax+1))
    P[0, 0] = 1
    for m in range(0,lmax):
        P[m+1, m+1] = (2*m + 1) * sx * P[m,m]
        P[m+1, m]   = (2*m + 1) * x * P[m,m]
        
    for m in range(0, lmax-1):
        for l in range(m+2, lmax+1):
            P[l, m] = ((2*l-1)*x*P[l-1, m]-(l-1+m)*P[l-2,m])/(l-m)
    
    return P


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
