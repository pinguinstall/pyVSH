from pyVSH.getLegendreP import getAB, getBetas, getCosSinMap, lm2idx
import numpy as np
from pyVSH.vshHelpers import getNumCoefficients

def getGlmFlm(maxl, a, d):
    """returns the line of gradients for a specific position on the sky"""
    flm = [0.0 for i in range(0, getNumCoefficients(maxl) // 2)]
    glm = [0.0 for i in range(0, getNumCoefficients(maxl) // 2)]

    AB = getAB(np.sin(d), maxl)
    A = AB[0]
    B = AB[1]
    betas = getBetas(maxl)
    cosmap, sinmap = getCosSinMap(a, maxl)

    i = 0
    j = 1 # was lm2idx(l, m) ... but if loop written in this way its just increment
    for l in range(1, maxl + 1):
        flm[i] = betas[j] * A[j] * cosmap[0]
        glm[i] = -betas[j] * B[j] * sinmap[0]
        i += 1
        j += 1
        for m in range(1, l + 1):
            flm[i] = 2 * betas[j] * A[j] * cosmap[m]
            glm[i] = -2 * betas[j] * B[j] * sinmap[m]
            i += 1

            flm[i] = -2 * betas[j] * A[j] * sinmap[m]
            glm[i] = -2 * betas[j] * B[j] * cosmap[m]
            i += 1
            j += 1

    return glm, flm


def getVecAlpha(F, G):
    return np.array([]).flatten()


def getVSHVectors(alpha, delta, lmax):
    G, F = getGlmFlm(lmax, alpha, delta)
    G = np.array(G)
    F = np.array(F)
    
    P = np.array([F, G]).flatten()
    Q = np.array([-G, F]).flatten()
    # P = for e alpha
    # Q = for e delta
    return np.transpose(np.array([P, Q]))
