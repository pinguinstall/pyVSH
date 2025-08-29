import numpy as np

def computeAlpha(lmax):
    alpha = np.zeros((lmax+1, lmax+1));
    fourpi = 4*np.pi;
    for l in range(0, lmax+1):
        alpha[l, 0] = np.sqrt((2*l+1)/fourpi)
        for m in range(1, l+1):
              alpha[l, m] = -alpha[l, m-1]/np.sqrt((l+m)*(l+1-m))

    return alpha
