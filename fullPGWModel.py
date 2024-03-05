# This file is part of pyMathHelpers.
# pyMathHelpers is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
# version.
# pyMathHelpers is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with pyMathHelpers. If not, see
# <https://www.gnu.org/licenses/>.

import numpy as np


class fullPGWModel:
    agw = 0.0
    dgw = 0.0
    hpluscos = 0.0
    hplussin = 0.0
    htimescos = 0.0
    htimessin = 0.0
    omgw = 0.0
    ePlus = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    eTimes = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    P = []
    p = []
    PePlusPt = []
    PeTimesPt = []

    def __init__(self, conf):
        self.agw = conf['agw']
        self.dgw = conf['dgw']
        self.hpluscos = conf['hpluscos']
        self.hplussin = conf['hplussin']
        self.htimescos = conf['htimescos']
        self.htimessin = conf['htimessin']
        self.omgw = conf['omgw']

        self.P = np.array(
            [[-np.sin(self.agw), -np.cos(self.agw) * np.sin(self.dgw), np.cos(self.agw) * np.cos(self.dgw)],
             [np.cos(self.agw), -np.sin(self.agw) * np.sin(self.dgw), np.sin(self.agw) * np.cos(self.dgw)],
             [0, np.cos(self.dgw), np.sin(self.dgw)]])

        self.p = np.array([np.cos(self.agw) * np.cos(self.dgw), np.sin(self.agw) * np.cos(self.dgw), np.sin(self.dgw)])

        self.PePlusPt = np.matmul(np.matmul(self.P, self.ePlus), np.transpose(self.P.copy()))
        self.PeTimesPt = np.matmul(np.matmul(self.P, self.eTimes), np.transpose(self.P.copy()))


    def getPGWShift(self, t, u):
        u = np.asarray(u)

        plusPhase = self.hpluscos * np.cos(self.omgw * t) + self.hplussin * np.sin(self.omgw * t)
        timesPhase = self.htimescos * np.cos(self.omgw * t) + self.htimessin * np.sin(self.omgw * t)

        PePlusPtloc = self.PePlusPt.copy() * plusPhase
        PeTimesPtloc = self.PeTimesPt.copy() * timesPhase

        h = PePlusPtloc + PeTimesPtloc
        s = u + self.p
        aas = np.sqrt(np.sum(np.array(s) ** 2))
        hu = -0.5 * np.matmul(h, u)

        if aas == 0:
            return hu

        s = s / aas
        hs = np.matmul(h, s)
        huu = s * (0.5 * aas * (aas * np.dot(hs, s) - 2 * np.dot(hs, self.p)) / np.dot(s, self.p))

        return huu + hu

    def getPGWShiftVectorized(self, t, u):
        N = len(t)
        u = np.asarray(u)

        plusPhase = self.hpluscos * np.cos(self.omgw * t) + self.hplussin * np.sin(self.omgw * t)
        timesPhase = self.htimescos * np.cos(self.omgw * t) + self.htimessin * np.sin(self.omgw * t)

        PePt = np.outer(self.PePlusPt.flatten(), plusPhase).T.reshape(N, 3, 3) + np.outer(self.PeTimesPt.flatten(),
                                                                                          timesPhase).T.reshape(N, 3, 3)

        huu = np.einsum('li, li->l', np.einsum('lij, lj->li', PePt, u), u)
        hu = 0.5 * np.einsum('lij, lj->li', PePt, u)

        return (((u + self.p).T / (2 * (1 + np.einsum('li, i -> l', u, self.p)))) * huu).T - hu
