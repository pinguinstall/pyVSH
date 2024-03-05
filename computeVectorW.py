# This file is part of pyMathHelpers.
# pyMathHelpers is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
# version.
# pyMathHelpers is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with pyMathHelpers. If not, see
# <https://www.gnu.org/licenses/>.

"""
provides functions for normalized powers in accordance with:
[M+K 12]:
Mignard, F., Klioner, S.;
Analysis of astrometric catalogues with vector spherical harmonics;
Astronomy & Astrophysics, Volume 547, id.A59, 18 pp.;
November 2012;
https://ui.adsabs.harvard.edu/link_gateway/2012A&A...547A..59M/doi:10.1051/0004-6361/201219927

This is a straight forward reference implementation with consistency with [M+K 12] in mind.
The code is not optimized for performance at this time.
"""

import numpy as np
import computeVectorP as Pl


def computeVectorW(lmax: int, vshCoeffs: np.array, vshSigmas: np.array) -> np.array:
    """
    Implements Equation (83) of [M+K 12] for all l up to lmax of a given set of VSH coefficients.
    Simple normalized powers per VSH order l.
    Normalization is done by sigma_l0 for P_l, sigma_lm with m > 0 is ignored.
    The VSH coefficients must be in the usual order 10R, 11R, 11I, 20R, 21R, 21I, 22R, 22I, ...
    :param lmax: maximum l (order of VSH expansion)
    :param vshCoeffs: VSH coefficients up to order l
    :param vshSigmas: corresponding (formal) standard errors
    :return: vector W_1, W_2, ..., W_lmax of normalized powers
    """
    i = 0
    nul = 0
    Wl = np.zeros(lmax)
    for l in range(1, lmax + 1):
        for m in range(0, l + 1):
            if m == 0:
                nul = i
                Wl[l - 1] += (vshCoeffs[i] / vshSigmas[nul]) ** 2
                i += 1
            else:
                Wl[l - 1] += (vshCoeffs[i] ** 2) / ((vshSigmas[nul] ** 2) / 2)
                i += 1
                Wl[l - 1] += (vshCoeffs[i] ** 2) / ((vshSigmas[nul] ** 2) / 2)
                i += 1
    return Wl


def computeVectorW_SH(lmax: int, shCoeffs: np.array, shSigmas: np.array) -> np.array:
    """
    Implements Equation (83) of [M+K 12] for all l up to lmax of a given set of VSH coefficients.
    Simple normalized powers per SH order l.
    Normalization is done by sigma_l0 for P_l, sigma_lm with m > 0 is ignored.
    The VSH coefficients must be in the usual order 10R, 11R, 11I, 20R, 21R, 21I, 22R, 22I, ...
    :param lmax: maximum l (order of VSH expansion)
    :param vshCoeffs: VSH coefficients up to order l
    :param vshSigmas: corresponding (formal) standard errors
    :return: vector W_1, W_2, ..., W_lmax of normalized powers
    """
    i = 0
    nul = 0
    Wl = np.zeros(lmax)
    for l in range(0, lmax + 1):
        for m in range(0, l + 1):
            if m == 0:
                nul = i
                Wl[l] += (vshCoeffs[i] / vshSigmas[nul]) ** 2
                i += 1
            else:
                Wl[l] += (vshCoeffs[i] ** 2) / ((vshSigmas[nul] ** 2) / 2)
                i += 1
                Wl[l] += (vshCoeffs[i] ** 2) / ((vshSigmas[nul] ** 2) / 2)
                i += 1
    return Wl


def computeVectorWOver(lmax: int, vshCoeffs: np.array, vshSigmas: np.array) -> np.array:
    """
    Implements Equation (87) of [M+K 12] for all l up to lmax of a given set of VSH coefficients.
    Normalized powers per VSH order l using average sigma_lm^2.
    The VSH coefficients must be in the usual order 10R, 11R, 11I, 20R, 21R, 21I, 22R, 22I, ...
    :param lmax: maximum l (order of VSH expansion)
    :param vshCoeffs: VSH coefficients up to order l
    :param vshSigmas: corresponding (formal) standard errors
    :return: vector WOverline_1, WOverline_2, ..., WOverline_lmax of normalized powers
    """
    sAvg = np.zeros(lmax)
    i = 0
    for l in range(1, lmax + 1):
        for m in range(0, l + 1):
            if m == 0:
                sAvg[l - 1] += vshSigmas[i] ** 2
                i += 1
            else:
                sAvg[l - 1] += 2 * (vshSigmas[i] ** 2)
                i += 1
                sAvg[l - 1] += 2 * (vshSigmas[i] ** 2)
                i += 1

    pl = Pl.computeVectorP(lmax, vshCoeffs)
    return np.array([pl[l - 1] / (sAvg[l - 1] / (2 * l + 1)) for l in range(1, lmax + 1)])


def computeVectorWOver_SH(lmax: int, shCoeffs: np.array, shSigmas: np.array) -> np.array:
    """
    Implements Equation (87) of [M+K 12] for all l up to lmax of a given set of VSH coefficients.
    Normalized powers per VSH order l using average sigma_lm^2.
    The VSH coefficients must be in the usual order 10R, 11R, 11I, 20R, 21R, 21I, 22R, 22I, ...
    :param lmax: maximum l (order of VSH expansion)
    :param vshCoeffs: VSH coefficients up to order l
    :param vshSigmas: corresponding (formal) standard errors
    :return: vector WOverline_1, WOverline_2, ..., WOverline_lmax of normalized powers
    """
    sAvg = np.zeros(lmax)
    i = 0
    for l in range(0, lmax + 1):
        for m in range(0, l + 1):
            if m == 0:
                sAvg[l] += vshSigmas[i] ** 2
                i += 1
            else:
                sAvg[l] += 2 * (vshSigmas[i] ** 2)
                i += 1
                sAvg[l] += 2 * (vshSigmas[i] ** 2)
                i += 1

    pl = Pl.computeVectorP(lmax, vshCoeffs)
    return np.array([pl[l] / (sAvg[l] / (2 * l + 1)) for l in range(0, lmax + 1)])


def computeVectorWTilde(lmax: int, vshCoeffs: np.array, vshSigmas: np.array) -> np.array:
    """
    Implements Equation (86) of [M+K 12] for all l up to lmax of a given set of VSH coefficients.
    Normalized powers per VSH order l using sigma_lm for each coefficient r_lm.
    The VSH coefficients must be in the usual order 10R, 11R, 11I, 20R, 21R, 21I, 22R, 22I, ...
    :param lmax: maximum l (order of VSH expansion)
    :param vshCoeffs: VSH coefficients up to order l
    :param vshSigmas: corresponding (formal) standard errors
    :return: vector WTilde_1, WTilde_2, ..., WTilde_lmax of normalized powers
    """
    Wtilde = np.zeros(lmax)
    i = 0
    for l in range(1, lmax + 1):
        for m in range(0, l + 1):
            if m == 0:
                Wtilde[l - 1] += (vshCoeffs[i] / vshSigmas[i]) ** 2
                i += 1
            else:
                Wtilde[l - 1] += (vshCoeffs[i] / vshSigmas[i]) ** 2
                i += 1
                Wtilde[l - 1] += (vshCoeffs[i] / vshSigmas[i]) ** 2
                i += 1
    return Wtilde


def computeVectorWTilde_SH(lmax: int, shCoeffs: np.array, shSigmas: np.array) -> np.array:
    """
    Implements Equation (86) of [M+K 12] for all l up to lmax of a given set of VSH coefficients.
    Normalized powers per VSH order l using sigma_lm for each coefficient r_lm.
    The VSH coefficients must be in the usual order 10R, 11R, 11I, 20R, 21R, 21I, 22R, 22I, ...
    :param lmax: maximum l (order of VSH expansion)
    :param vshCoeffs: VSH coefficients up to order l
    :param vshSigmas: corresponding (formal) standard errors
    :return: vector WTilde_1, WTilde_2, ..., WTilde_lmax of normalized powers
    """
    Wtilde = np.zeros(lmax)
    i = 0
    for l in range(0, lmax + 1):
        for m in range(0, l + 1):
            if m == 0:
                Wtilde[l] += (vshCoeffs[i] / vshSigmas[i]) ** 2
                i += 1
            else:
                Wtilde[l] += (vshCoeffs[i] / vshSigmas[i]) ** 2
                i += 1
                Wtilde[l] += (vshCoeffs[i] / vshSigmas[i]) ** 2
                i += 1
                
    return Wtilde
