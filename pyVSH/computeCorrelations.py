# This file is part of pyMathHelpers.
# pyMathHelpers is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
# version.
# pyMathHelpers is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with pyMathHelpers. If not, see
# <https://www.gnu.org/licenses/>.

import numpy as np


def computeCorrelations(inv_mat):
    """
    compute the correlations from a inverse normals matrix from an LSQ problem
    :param inv_mat: inverted matrix
    :return: correlation matrix
    """
    sigmas = np.sqrt(np.diag(inv_mat))
    return inv_mat / np.outer(sigmas, sigmas)
