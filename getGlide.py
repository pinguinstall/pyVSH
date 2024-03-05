# This file is part of pyMathHelpers.
# pyMathHelpers is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
# version.
# pyMathHelpers is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with pyMathHelpers. If not, see
# <https://www.gnu.org/licenses/>.

import numpy as np


def getGlide(res):
    """
    compute the glide from the results of a VSH fit.
    definitions and notations are in line with
    https://ui.adsabs.harvard.edu/link_gateway/2012A&A...547A..59M/doi:10.1051/0004-6361/201219927

    :param res: result dict from VSH fit (see fitVSH.fitVSH())
    :return: col 0 glide [G1, G2, G3], col 1 formal uncertainties of glide [sigma1, sigma2, sigma3]
    """
    p = -np.sqrt(3 / (4 * np.pi))
    q = np.sqrt(3 / (8 * np.pi))
    half = int(res['solution'].shape[0] / 2)

    return np.array([[p * res['solution'][half + 1], abs(p) * res['sigmas'][half + 1]],
                     [-p * res['solution'][half + 2], abs(p) * res['sigmas'][half + 2]],
                     [q * res['solution'][half], q * res['sigmas'][half]]])
