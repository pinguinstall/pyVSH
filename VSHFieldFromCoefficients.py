import getVSHDerivatives as vshd
from celestCoordinates import getUHValpha, getUHVdelta, vec2ad
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

    uvalpha = getUHValpha(alpha)
    uvdelta = getUHVdelta(alpha, delta)

    sigAlphaStar = np.dot(v[:, 0], vshcofs)
    sigDelta = np.dot(v[:, 1], vshcofs)
    sigVec = sigAlphaStar * uvalpha + sigDelta * uvdelta

    return [sigAlphaStar, sigDelta, sigVec[0], sigVec[1], sigVec[2]]



def plotAlphaDeltaForCoffSet(cofs, nside=16,
                             savenames=[],
                             titles=[r"in $\mathbf{e}_\alpha$", r"in $\mathbf{e}_\delta$", r"$overall$"],
                             cbarlabels=["","",""],
                             dpi=100):
    hpCenterVecs = np.array([hp.pix2vec(nside, i) for i in range(0, hp.nside2npix(nside))])
    hpCenterAD = np.array([vec2ad(v) for v in hpCenterVecs])
    hpcenterVSH = np.array([getVSHFieldForPos(cofs, ad[0], ad[1]) for ad in hpCenterAD])

    maxVal = np.max(np.abs(hpcenterVSH))

    malpha = hp.ma(hpcenterVSH[:, 0])
    mdelta = hp.ma(hpcenterVSH[:, 1])
    mall = hpcenterVSH[:, [2,3,4]]
    mall = np.sum(np.abs(mall) ** 2, axis=-1) ** (1. / 2)

    pixel_theta, pixel_phi = hp.pix2ang(nside, [int(h) for h in range(0, hp.nside2npix(nside))])

    hp.mollview(malpha, cmap='seismic', min=-maxVal, max=maxVal, xsize=3000, title=titles[0])
    plt.gca().images[-1].colorbar.set_label(cbarlabels[0])
    if len(savenames) == 3:
        plt.savefig(savenames[0], dpi=dpi, bbox_inches='tight')
        
    hp.mollview(mdelta, cmap='seismic', min=-maxVal, max=maxVal, xsize=3000, title=titles[1])
    plt.gca().images[-1].colorbar.set_label(cbarlabels[1])
    if len(savenames) == 3:
        plt.savefig(savenames[1], dpi=dpi, bbox_inches='tight')
        
    hp.mollview(mall, cmap='jet', min=0, max=np.max(mall), xsize=3000, title=titles[2])
    plt.gca().images[-1].colorbar.set_label(cbarlabels[2])
    if len(savenames) == 3:
        plt.savefig(savenames[2], dpi=dpi, bbox_inches='tight')
