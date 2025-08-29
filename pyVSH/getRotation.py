import numpy as np

def getRotation(res):
    p = -np.sqrt(3 / (4 * np.pi))
    q = np.sqrt(3 / (8 * np.pi))

    return np.array([[ p*res['solution'][1], abs(p)*res['sigmas'][1]],
                     [-p*res['solution'][2], abs(p)*res['sigmas'][2]],
                     [ q*res['solution'][0], q*res['sigmas'][0]]])