import getVSHDerivatives as vshd
from celestCoordinates import getUHValpha, getUHVdelta, vec2ad, vec2adVectorized
from vshHelpers import getLmaxFromNumCo
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

def getVSHFieldForPos(vshcofs, alpha, delta):
    """
    returns the local vector field components according to the VSH coefficients
    :param vshcofs: vector of VSH coefficients
    :param alpha: RA
    :param delta: DEC
    :return: vector field components as [sigAlphaStar, sigDelta, sigVecX, sigVecY, sigVecZ]
    """
    numCoeffs = np.array(vshcofs).shape[0]
    numCoeffsHalf = numCoeffs // 2

    lmax = getLmaxFromNumCo(numCoeffs)
    v = vshd.getVSHVectors(alpha, delta, lmax)
    #print(v)
    uvalpha = getUHValpha(alpha)
    uvdelta = getUHVdelta(alpha, delta)

    sigAlphaStar = np.dot(v[:, 0], vshcofs)
    sigDelta = np.dot(v[:, 1], vshcofs)
    sigVec = sigAlphaStar * uvalpha + sigDelta * uvdelta

    return [sigAlphaStar, sigDelta, sigVec[0], sigVec[1], sigVec[2]]


def plotAlphaDeltaForCoffSet(cofs, nside=32,
                             savenames=[],
                             titles=[r"in $\mathbf{e}_\alpha$", r"in $\mathbf{e}_\delta$", r"$overall$"],
                             cbarlabels=["","",""],
                             dpi=160,
                             legendextend="neither",
                             markers=None):
    hpCenterVecs = np.array(hp.pix2vec(nside, range(0, hp.nside2npix(nside)))).T
    hpCenterAD = vec2adVectorized(hpCenterVecs)
    hpcenterVSH = np.array([getVSHFieldForPos(cofs, ad[0], ad[1]) for ad in hpCenterAD])

    maxVal = np.max(np.abs(hpcenterVSH))

    malpha = hp.ma(hpcenterVSH[:, 0])
    mdelta = hp.ma(hpcenterVSH[:, 1])
    # mall = hpcenterVSH[:, [2, 3, 4]]
    # mall = np.sqrt(np.sum(mall**2, axis=1))
    mall = np.sqrt(malpha**2 + mdelta**2)

    pixel_theta, pixel_phi = hp.pix2ang(nside, [int(h) for h in range(0, hp.nside2npix(nside))])


    hp.projview(malpha, cmap='seismic', min=-maxVal, max=maxVal,
                xsize=1000, title=titles[0],
                projection_type='aitoff', unit=cbarlabels[0], extend=legendextend)
    if len(savenames) == 3:
        plt.savefig(savenames[0], dpi=dpi, bbox_inches='tight')
        

    hp.projview(mdelta, cmap='seismic', min=-maxVal, max=maxVal,
                xsize=1000, title=titles[1],
                projection_type='aitoff', unit=cbarlabels[1], extend=legendextend)
    if len(savenames) == 3:
        plt.savefig(savenames[1], dpi=dpi, bbox_inches='tight')
        

    hp.projview(mall, cmap='jet', min=0, max=maxVal,
                xsize=1000, title=titles[2],
                projection_type='aitoff', unit=cbarlabels[2], extend=legendextend)
    if len(savenames) == 3:
        plt.savefig(savenames[2], dpi=dpi, bbox_inches='tight')
