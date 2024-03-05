# This file is part of pyMathHelpers.
# pyMathHelpers is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
# version.
# pyMathHelpers is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with pyMathHelpers. If not, see
# <https://www.gnu.org/licenses/>.

"""
provides function for un-normalized powers in accordance with:
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


def computeVectorP(lmax: int, vshCoeffs: np.array) -> np.array:
    """
    Implements Equation (76) of [M+K 12],
    computes the un-normalized powers P_l for a set of VSH coefficients (toroidal or spheroidal)
    up to order l_max
    :param lmax: maximum l up to which vshCoeffs are given
    :param vshCoeffs: the coefficients in the usual order
    :return: a vector with l=1, ..., lmax power values in it
    """
    i = 0
    Pl = np.zeros(lmax)
    for l in range(1, lmax + 1):
        Pl[l - 1] += vshCoeffs[i] ** 2
        i += 1
        for m in range(1, l + 1):
            Pl[l - 1] += 2 * vshCoeffs[i] ** 2
            i += 1
            Pl[l - 1] += 2 * vshCoeffs[i] ** 2
            i += 1
    return Pl


def computePl_SH(lmax: int, shCoeffs: np.array) -> np.array:
    """
    Implements Equation (76) of [M+K 12],
    computes the un-normalized powers P_l for a set of SH coefficients
    up to order l_max
    :param lmax: maximum l up to which shCoeffs are given
    :param shCoeffs: the coefficients in the usual order
    :return: a vector with l=1, ..., lmax power values in it
    """
    i = 0
    Pl = np.zeros(lmax+1)
    for l in range(0, lmax + 1):
        #print("l = {}, i = {}".format(l,i))
        Pl[l] += shCoeffs[i] ** 2
        i += 1
        for m in range(1, l + 1):
            #print("l = {}, i = {}".format(l,i))
            Pl[l] += 2 * shCoeffs[i] ** 2
            i += 1
            #print("l = {}, i = {}".format(l,i))
            Pl[l] += 2 * shCoeffs[i] ** 2
            i += 1
    return Pl
