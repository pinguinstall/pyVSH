# This file is part of pyMathHelpers.
# pyMathHelpers is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
# version.
# pyMathHelpers is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with pyMathHelpers. If not, see
# <https://www.gnu.org/licenses/>.

import getVSHDerivatives as vshd
import numpy as np
import getVFieldStats
from computeCorrelations import computeCorrelations
from getRotation import getRotation
from getGlide import getGlide
from computeVectorP import computeVectorP
import progressBar
from computeVectorW import computeVectorW, computeVectorWOver, computeVectorWTilde
import sys


# col 0 = alphas
# col 1 = deltas
# col 2 = signal in e alpha
# col 3 = sigma in e alpha
# col 4 = signal in e delta
# col 5 = sigma in e delta


def fitVSH(data: np.array, lmax: int, debug=False, store_all=False) -> tuple:
    """
    fit vector spherical harmonics up to order lmax to the input data.
    the data must be a matrix with the following columns
    # col 0 = alphas (position)
    # col 1 = deltas (position)
    # col 2 = signal in e alpha
    # col 3 = sigma in e alpha
    # col 4 = signal in e delta
    # col 5 = sigma in e delta

    :param data: data input matrix
    :param lmax: maximum order to fit to
    :param debug: print debug info (default False)
    :param store_all: store everything (incl. design matrices, residuals, etc) in the result,
    can make output very large
    :return: dict with all results
    """
    if debug:
        print("   setup & get coefficients", file=sys.stderr, flush=True)
    M = data.shape[0]
    res = {'datapoints': M, 'pwrVField': getVFieldStats.getPowerOfVectorField(data[:, [2, 4]])}
    num_coeffs = vshd.getNumCoefficients(lmax)
    num_coeffs_half = int(num_coeffs / 2)
    res['lmax'] = lmax

    A = np.zeros((2 * M, num_coeffs))
    b = np.zeros(2 * M)
    W = np.zeros(2 * M)

    b[:M] = data[:, 2]
    b[M:] = data[:, 4]

    # W stays 1/sigma here, because below we compute AW^T * AW and the get the power automatically
    W[:M] = (1 / np.array(data[:, 3]))
    W[M:] = (1 / np.array(data[:, 5]))

    pb = progressBar.progressBar(1, M, 60)
    for i in range(0, M):
        alpha = data[i, 0]
        delta = data[i, 1]
        v = vshd.getVSHVectors(alpha, delta, lmax)
        A[i, :] = v[:, 0]
        A[i + M, :] = v[:, 1]
        pb.getProgressIncrement()

    # [:, np.newaxis] because we want to multiply every row of A with the specific value in W
    AW = A * W[:, np.newaxis]
    bW = b * W

    if store_all:
        res['b'] = b.copy()
        res['W'] = W.copy()
        res['A'] = A.copy()

    if debug:
        print("   solve system", file=sys.stderr, flush=True)
    # solve the system
    N = np.matmul(np.transpose(AW), AW)
    rhs = np.matmul(np.transpose(AW), bW)
    res['AtA'] = N
    res['Atb'] = rhs
    c = np.linalg.solve(N, rhs)
    res['solution'] = c

    if debug:
        print("   SVD", file=sys.stderr, flush=True)
    # condition number via svd
    _, s, _ = np.linalg.svd(N)
    res['condNr'] = s[0] / s[-1]

    if debug:
        print("   inverse & correlations", file=sys.stderr, flush=True)
    # correlations
    invN = np.linalg.inv(N)
    corrs = computeCorrelations(invN)
    res['allCorrs'] = corrs.astype(np.half)
    res['maxCorr'] = np.max(np.triu(corrs, 1))
    res['minCorr'] = np.min(np.triu(corrs, 1))
    del corrs

    # uncertainties
    v = (np.matmul(A, c) - b)

    if store_all:
        res['residuals'] = v

    vw = v * W
    s02 = np.dot(vw, vw) / (2 * M - num_coeffs)
    sigmas = np.sqrt(np.diag(invN)) * np.sqrt(s02)
    res['uwe'] = np.sqrt(s02)
    res['sigmas'] = sigmas

    if debug:
        print("   rotation & glide, Ps & Ws", file=sys.stderr, flush=True)

    # rotation
    res['rot'] = getRotation(res)

    # glide
    if lmax > 0:
        res['glide'] = getGlide(res)
    else:
        res['glide'] = 'lmax == 0'

    # RMS and Powers
    PlT = computeVectorP(lmax, c[:num_coeffs_half])
    PlS = computeVectorP(lmax, c[num_coeffs_half:])
    res['PlT'] = PlT.copy()
    res['PlS'] = PlS.copy()

    RMSAll = np.sqrt((PlT + PlS) / (4 * np.pi))
    PlT = np.sqrt(PlT / (4 * np.pi))
    PlS = np.sqrt(PlS / (4 * np.pi))

    res['RMSVSH'] = np.transpose(np.array([PlT, PlS, RMSAll]))
    res['RMSData'] = np.sqrt(np.dot(b, b) / (2 * M))
    res['RMSResult'] = np.sqrt(s02 * (2 * M - num_coeffs) / (2 * M))

    # normalized powers
    res['WlT'] = computeVectorW(lmax, c[:num_coeffs_half], sigmas[:num_coeffs_half])
    res['WlS'] = computeVectorW(lmax, c[num_coeffs_half:], sigmas[num_coeffs_half:])
    res['WlTildeT'] = computeVectorWTilde(lmax, c[:num_coeffs_half], sigmas[:num_coeffs_half])
    res['WlTildeS'] = computeVectorWTilde(lmax, c[num_coeffs_half:], sigmas[num_coeffs_half:])
    res['WlOverT'] = computeVectorWOver(lmax, c[:num_coeffs_half], sigmas[:num_coeffs_half])
    res['WlOverS'] = computeVectorWOver(lmax, c[num_coeffs_half:], sigmas[num_coeffs_half:])

    return res
