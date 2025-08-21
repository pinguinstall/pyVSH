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


def lmReIm2idx(l, m, ri="re"):
    """
    Gives the flat index of VSH coefficients in the usual order 10R, 11R, 11I, 20R, 21R, 21I, 22R, 22I, ...
    :param l: l
    :param m: m
    :param ri: Real or Imaginary part
    :return: flat index, None if invalid combination (e.g. m>l)
    """
    if (m == 0) and (ri == "im"):
        return None
    if m > l:
        return None
    offset = ((l-1) * ((l-1) + 2))

    if m == 0:
        return offset
    
    if ri == "im":
        i = 0
    elif ri == "re":
        i = 1
    else:
        return None
    
    return offset + 2*m - i
