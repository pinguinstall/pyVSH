import numpy as np

def computeCosSin(lamb, lmax):
    cosMap = np.zeros(lmax+1)
    sinMap = np.zeros(lmax+1)
    cosMap[0] = 1
    c1 = np.cos(lamb)
    s1 = np.sin(lamb)
    cosMap[1] = c1
    sinMap[1] = s1
    for m in range(3, lmax+2):
        cosMap[m-1] = c1 * cosMap[m - 2] - s1 * sinMap[m - 2]
        sinMap[m-1] = s1 * cosMap[m - 2] + c1 * sinMap[m - 2]
    return np.array([cosMap, sinMap]).T
