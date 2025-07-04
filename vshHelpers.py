import numpy as np


def getLmaxFromNumCo(nc):
    return int(-1 + np.sqrt(1+nc/2))


def getNumCoefficients(maxl):
    return 2 * (maxl * (maxl + 2))


def getNumCoefficientsForSingleL(l):
    return int(2 * l + 1)


def setAllExceptLToZero(vshcofs, l):
    numCoeffs = np.array(vshcofs).shape[0]
    numCoeffsHalf = int(numCoeffs / 2)
    lmax = getLmaxFromNumCo(numCoeffs)
    ncforl = getNumCoefficientsForSingleL(l)

    cofsnew = np.array(vshcofs).copy()

    slocs = np.array(range(int(getNumCoefficients(l-1)/2), int(getNumCoefficients(l)/2)))
    cofsnew[slocs] = 0.0
    slocs = numCoeffsHalf + slocs
    cofsnew[slocs] = 0.0

    return np.array(vshcofs) - cofsnew
