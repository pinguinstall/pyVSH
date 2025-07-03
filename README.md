# pyVSH
**`pyVSH`** is a library allowing you to fit spherical harmonics (SH) and vector spherical harmonics (VSH) to data, mainly astronomical catalogues.
The code is a more or less complete implementation of the SH and VSH dicussed in: 
Mignard, F. and Klioner, S.: "*Analysis of astrometric catalogues with vector spherical harmonics*" 
available under [https://ui.adsabs.harvard.edu/abs/2012A%26A...547A..59M/abstract].

**`pyVSH`** provides functions to compute the SH and VSH base functions, and create the design matrices to fit the SH and VSH coefficents for
arbitrary data using a simple weighted least squares. The library uses recurrence formulas to compute the base functions avoiding numerical
issues for higher orders (l).

We also included functionality to compute weighted and un-weighted powers from the fitted coefficients, rotation and glide, and some other parameters.

## usage
In the simplest case you have some tabular data, with rows wise (for fitting SH):
> `right ascension [rad]; declination [rad]; signal; uncertainty of signal` ,

or row by row, for VSH:
> `right asc [rad]; decl [rad]; signal RA; uncertainty RA; signal DEC; uncertainty DEC` .

Make sure this data is in a numpy array, and you just need to call `fitVSH.fitVSH(data, lmax)`.

Here is a short example for random data:
```python
import numpy as np
import fitVSH
n_points = 3000

# generate randomly distributed points on the sphere
alphas = np.random.rand(n_points) * 2.0 * np.pi
deltas = np.arcsin(np.random.rand(n_points) * 2.0 - 1.0)

# generate random, white noise, data
# first randomly chose the uncertainties to use
sigmaalpha = np.random.rand(n_points) * 2.0 + 1 # gives sigma 1 to 3
sigmadelta = np.random.rand(n_points) * 2.0 + 1 # gives sigma 1 to 3

# second generate white noise data according to the sigmas
dalpha = np.random.randn(n_points) * sigmaalpha
ddelta = np.random.randn(n_points) * sigmadelta

# pack it all together
mydata = np.array([alphas, deltas, dalpha, sigmaalpha, ddelta, sigmadelta]).T

# fit VSH up to order lmax
lmax = 5
result = fitVSH.fitVSH(mydata, lmax=lmax)
```

The result is a dictionary with the usual statistics and fit parameters.
