# This file is part of pyMathHelpers.
# pyMathHelpers is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
# version.
# pyMathHelpers is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with pyMathHelpers. If not, see
# <https://www.gnu.org/licenses/>.

import numpy as np


def vectorAngle(vec1, vec2):
    """
    Computes the angle between the vectors in a numerically stable way.
    Input are two array like vectors or list of vectors.
    Output is a scalar if only a single pair of vectors is inputed, 
    for multiple input vectors the result is a flat array of angles.
        Parameters:
                    vec1 (list like): a single vector or a list or array of vectors
                    vec2 (list like): a single vector or a list or array of vectors

        Returns:
                    angles (array or scalar): angles between vectors in vec1 and vectors in vec2
    """
    v1 = np.atleast_2d(np.asarray(vec1))
    v2 = np.atleast_2d(np.asarray(vec2))

    # compute norms of all vectors ||v||
    l1 = np.einsum("ij,ij->i", v1, v1)
    l2 = np.einsum("ij,ij->i", v2, v2)
    lm = np.sqrt(np.maximum(l1,l2)) # pair wise maximum

    # scale each vector by the maximum scale
    v1s = v1 / lm[:,None]
    v2s = v2 / lm[:,None]

    # compute dot product between each vector
    dots = np.einsum("ij,ij->i", v1s, v2s)

    # compute cross product between each vector
    crosses = np.cross(v1s, v2s)

    # compute norm of crosses
    cs = np.sqrt(np.sum(crosses**2, axis=1))

    res = np.arctan2(cs, dots)
    if res.shape[0] == 1:
        return res[0]
    return res


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

        if (u == self.p).all():
            return np.array([0.0, 0.0, 0.0])
        if (u == -self.p).all():
            return np.array([0.0, 0.0, 0.0])

        plusPhase = self.hpluscos * np.cos(self.omgw * t) + self.hplussin * np.sin(self.omgw * t)
        timesPhase = self.htimescos * np.cos(self.omgw * t) + self.htimessin * np.sin(self.omgw * t)

        PePlusPtloc = self.PePlusPt.copy() * plusPhase # p^+_ij (eq 3)
        PeTimesPtloc = self.PeTimesPt.copy() * timesPhase # p^x_ij (eq 4)

        h = PePlusPtloc + PeTimesPtloc
        s = u + self.p # u^i + p^i in (eq 1)
        aas = np.sqrt(np.sum(np.array(s) ** 2))
        s = s / aas
        if aas == 0:
            return hu

        hu = -0.5 * np.matmul(h, u) # second part of (eq 1)
        hs = np.matmul(h, s)
        huu = s * (0.5 * aas * (aas * np.dot(hs, s) - 2 * np.dot(hs, self.p)) / np.dot(s, self.p))

        return huu + hu


    def getPGWShiftVectorized(self, t, u):
        N = len(t)
        u = np.asarray(u)
        t = np.asarray(t)

        plusPhase = self.hpluscos * np.cos(self.omgw * t) + self.hplussin * np.sin(self.omgw * t)
        timesPhase = self.htimescos * np.cos(self.omgw * t) + self.htimessin * np.sin(self.omgw * t)

        PePt = np.outer(self.PePlusPt.flatten(), plusPhase).T.reshape(N, 3, 3) + np.outer(self.PeTimesPt.flatten(),
                                                                                          timesPhase).T.reshape(N, 3, 3)

        huu = np.einsum('li, li->l', np.einsum('lij, lj->li', PePt, u), u)
        hu = 0.5 * np.einsum('lij, lj->li', PePt, u)

        return np.nan_to_num((((u + self.p).T / (2 * (1 + np.einsum('li, i -> l', u, self.p)))) * huu).T - hu, nan=0.0, copy=False)


    def getSourceTerm(self, t0, u, x0 = None):
        return np.array([0, 0, 0])


    def getPGWShiftFromEllipsisModel(self, t, u):
        """
        computes the same as getPGWShift() for a single time and a single u but using the explicit
        ellipsis formulas from Geyer 2024 Appendix A. These routines can be used to check other code above.
        :param t: time of observation
        :param u: unit vector towards observed astrometrical source
        :return: delta u, the GW signal perpenticular to observation direction vector
        """
        alphaGW = self.agw
        deltaGW = self.dgw
        alpha, delta = vec2ad(u)
        triad = getLocalTriad(alpha, delta) # row wise e_a, e_d, r
        theta = vectorAngle(self.p, u)
        calN = 4 * np.sin(theta) * (1 + np.cos(theta))
        calF = np.cos(delta)*np.sin(2*(alpha - alphaGW))*(-3 + np.cos(2*deltaGW) - 4*np.sin(delta)*np.sin(deltaGW)) - 4*np.cos(deltaGW)*np.sin(alpha - alphaGW)*(np.cos(2*delta) - np.sin(delta)*np.sin(deltaGW))
        calG = 3*(np.cos(deltaGW)**2)*np.sin(2*delta) + 4*np.cos(alpha - alphaGW)*np.cos(deltaGW)*(np.sin(delta) - np.cos(2*delta)*np.sin(deltaGW)) - 2*np.cos(2*(alpha - alphaGW))*np.cos(delta)*(2*np.sin(deltaGW) + np.sin(delta)*(1 + (np.sin(deltaGW)**2)))
        D = (0.5 * np.sin(theta)) * (1 / calN) * np.array([[calF, -calG], [calG, calF]])
        phi = self.omgw * t
        b = np.array([self.hplussin * np.sin(phi) + self.hpluscos * np.cos(phi), self.htimessin * np.sin(phi) + self.htimescos * np.cos(phi)])
        Db = D@b
        return Db[0] * triad[0] + Db[1] * triad[1]


    def getPerturbedAngleBetweenTwoStars(self, u_0, u_1, t):
        plusPhase = self.hpluscos * np.cos(self.omgw * t) + self.hplussin * np.sin(self.omgw * t)
        timesPhase = self.htimescos * np.cos(self.omgw * t) + self.htimessin * np.sin(self.omgw * t)

        PePlusPtloc = self.PePlusPt.copy() * plusPhase # p^+_ij (eq 3)
        PeTimesPtloc = self.PeTimesPt.copy() * timesPhase # p^x_ij (eq 4)

        h = PePlusPtloc + PeTimesPtloc
        
        k1 = -np.asarray(u_0).copy()
        k2 = -np.asarray(u_1).copy()
        
        k1k2 = k1@k2
        kkkp1 = (k1k2 - k2@(self.p))/(1 - k1@(self.p))
        kkkp2 = (k1k2 - k1@(self.p))/(1 - k2@(self.p))
        kkpkkp = np.einsum('j,k', k1, k1) * kkkp1 + np.einsum('j,k', k2, k2) * kkkp2
        hijkkpkkp = 0.5 * np.einsum('jk,jk', h, kkpkkp)
        
        hijkk = np.einsum('jk,j,k', h, k1, k2)
        
        return k1k2 + hijkkpkkp - hijkk

    
    def getDeltaMax(self):
        """
        return maximum effect produced by given GW given the 4 amplitudes which also encode the phase
        :param htimescos: htimescos
        :param hplussin: htimescos
        :param htimessin: htimessin
        :param hpluscos: hpluscos
        :return: delta max
        """
        D = self.htimescos * self.hplussin - self.htimessin * self.hpluscos
        h = np.sqrt(self.htimescos**2 +  self.hplussin**2 + self.htimessin**2 + self.hpluscos**2)
        #clip = lambda x, threshold=1e-15: 0 if abs(x) < threshold else x
        S = np.sqrt((1 - (4 * D**2)/(h**4)))
        return 0.5 * (1/np.sqrt(2)) * h * np.sqrt(1 + S)


    def getEccentricity(self):
        """
        returns the eccentricity of the GW effect given its parameters
        :return: e (0 -> 1)
        """
        D = self.htimescos * self.hplussin - self.htimessin * self.hpluscos
        h = np.sqrt(self.htimescos**2 +  self.hplussin**2 + self.htimessin**2 + self.hpluscos**2)
        #clip = lambda x, threshold=1e-15: 0 if abs(x) < threshold else x
        S = np.sqrt((1 - (4 * D**2)/(h**4)))
        return np.sqrt((2 * S)/(1 + S))

    def getPhiAngle(self):
        """
        returns the ellipsis position angle
        :return: phi (0 -> 2pi)
        """
        hplussq  = self.hpluscos**2 + self.hplussin**2
        htimessq = self.htimescos**2 + self.htimessin**2
        h = np.sqrt(hplussq + htimessq)
        
        B = self.htimescos * self.hpluscos + self.hplussin * self.htimessin
        if B == 0:
            if np.sqrt(htimessq) < np.sqrt(hplussq):
                return 0
            elif np.sqrt(htimessq) > np.sqrt(hplussq):
                return np.pi/2
        
        D = self.htimescos * self.hplussin - self.htimessin * self.hpluscos
        #clip = lambda x, threshold=1e-15: 0 if abs(x) < threshold else x
        S = np.sqrt((1 - (4 * D**2)/(h**4)))
        
        return np.arctan((-hplussq + htimessq + S*h**2)/(2 * B))


    def getTheta(self, u):
        """
        returns the angle between the GW propagation direction and a given u
        :return: theta (0 -> pi)
        """
        return vectorAngle(self.p, u)
