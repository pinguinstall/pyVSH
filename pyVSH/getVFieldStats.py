import numpy as np

def getPowerOfVectorField(xyData):
    s = np.sum([np.sum(d**2) for d in xyData])
    return (4*np.pi/xyData.shape[0]) * s
