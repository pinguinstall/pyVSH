# This file is part of pyMathHelpers.
# pyMathHelpers is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
# version.
# pyMathHelpers is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with pyMathHelpers. If not, see
# <https://www.gnu.org/licenses/>.

import numpy as np


def ad2vec(alpha, delta):
    """
    converts RA DEC celestial coordinates to a unit vector pointing to this coordinates
    :param alpha: celestial right ascension
    :param delta: celestial declination
    :return: unit vector pointing to the direction
    """
    return np.array([np.cos(alpha) * np.cos(delta), np.sin(alpha) * np.cos(delta), np.sin(delta)])


def ad2vecVectorized(advec):
    """
    converts RA DEC celestial coordinates to unit vectors pointing to these coordinates.
    vectorized version for multiple sets of parameters
    :param advec: maxtrix of pairs of [[alpha_1, delta_1], [alpha_2, delta_2], ...]
    :return: unit vectors pointing to the direction (maxtrix [[x_1,y_1,z_1], [x_2,y_2,z_2], ...])
    """
    ca = np.array(np.cos(advec[:, 0]))
    sa = np.array(np.sin(advec[:, 0]))
    cd = np.array(np.cos(advec[:, 1]))
    sd = np.array(np.sin(advec[:, 1]))
    return np.transpose(np.array([ca * cd, sa * cd, sd]))


def ad2vecArr(ad):
    """
    wrapper for ad2vec, accepts an array [alpha, delta]
    :param ad: an array [alpha, delta]
    :return: unit vector pointing to the direction
    """
    return ad2vec(ad[0], ad[1])


def getUHValpha(alpha):
    """
    returns the unit vector in the local direction of increasing right ascension.
    this only depends on local alpha
    :param alpha: right ascension
    :return: vector e_alpha
    """
    return np.array([-np.sin(alpha), np.cos(alpha), 0.0])


def getUHVdelta(alpha, delta):
    """
    returns the unit vector in the local direction of increasing declination.
    :param alpha: right ascension
    :param delta: declination
    :return: vector e_delta
    """
    return np.array([-np.cos(alpha) * np.sin(delta), -np.sin(alpha) * np.sin(delta), np.cos(delta)])


def getLocalTriad(alpha, delta):
    """
    returns the local triad, e_alpha, e_delta, r
    unit vector in the direction of increasing RA
    unit vector in the direction of increasing DEC
    unit vector towards the point
    :param alpha: RA
    :param delta: DEC
    :return: matrix with row wise e_a, e_d, r
    """
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cd = np.cos(delta)
    sd = np.sin(delta)

    return np.array([[-sa, ca, 0.],
                     [-sd * ca, -sd * sa, cd],
                     [cd * ca, cd * sa, sd]])


def getLocalTriadVectorized(alphas, deltas):
    """
    returns the local triad, e_alpha, e_delta, r
    unit vector in the direction of increasing RA
    unit vector in the direction of increasing DEC
    unit vector towards the point
    :param alphas: list of RA
    :param deltas: list DEC
    :return: list with with row wise e_a, e_d, r
    """
    alphas = np.reshape(alphas, (-1,))
    deltas = np.reshape(deltas, (-1,))
    
    ca = np.cos(alphas)
    sa = np.sin(alphas)
    cd = np.cos(deltas)
    sd = np.sin(deltas)

    return {"eas" : np.array([-sa, ca, np.zeros(alphas.shape[0])]).T, "eds" : np.array([-sd * ca, -sd * sa, cd]).T, "rs" : np.array([cd * ca, cd * sa, sd]).T}


def projectToUValphaDelta(sig_vec, alpha, delta):
    """
    projects the signal given as vector on the unit sphere to the
    unit vectors e_alpha, e_delta at the given celestial coordinate.
    no checks are performed, make sure sig_vec is a tangential vector at
    (alpha, delta) and a unit vector.
    :param sig_vec: tangential vector in some arbitrary direction at alpha, delta
    :param alpha: right ascension
    :param delta: declination
    :return: tuple (sig_vec . ea), (sig_vec . ed), where "." is the dot product
    """
    ea = getUHValpha(alpha)
    ed = getUHVdelta(alpha, delta)
    return np.dot(sig_vec, ea), np.dot(sig_vec, ed)


def vec2ad(vec):
    """
    converts the unit vector vec to the corresponding celestial coordinates in RA and DEC
    :param vec: unit pointing vector
    :return: tuple (alpha, delta)
    """
    xy = np.sqrt(vec[0] ** 2 + vec[1] ** 2)
    if xy <= 1.0e-16:
        return 0.0, 0.5 * np.pi * np.sign(vec[2])
    a = np.arctan2(vec[1], vec[0])
    if a < 0:
        a += 2.0 * np.pi
    return a, np.arctan2(vec[2], xy)


def vec2adVectorized(vecs):
    """
    converts multiple unit vectors pointing to points on the unit sphere to
    their according RA and DEC values
    :param vecs: matrix of 3d vectors
    :return: matrix of [alpha, delta] per row
    """
    xy = np.sqrt(vecs[:, 0] ** 2 + vecs[:, 1] ** 2)
    a = np.arctan2(vecs[:, 1], vecs[:, 0])
    a = np.where(a < 0, a + 2.0 * np.pi, a)
    d = np.arctan2(vecs[:, 2], xy)
    
    a = np.where(xy <= 0.0, 0.0, a)
    d = np.where(xy <= 0.0, 0.5 * np.pi * np.sign(vecs[:, 2]), d)
    
    return np.array([a, d]).T
