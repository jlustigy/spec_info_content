from ext_func.rsky import rsky
from ext_func.occultquad import occultquad
import numpy as np
import math
import matplotlib.pyplot as plt

"""
Calculates a transit light curve assuming a
quadratic LD law:

I(mu) = 1 - u1*(1-mu) - u2*(1-mu)**2

Input:
e 	= eccentricity
i	= inclination angle (radians!)
u1 	= limb darkening parameter 1
u2 	= limb darkening parameter 2
p0 	= planet-to-star radius ratio
w	=
period	= planet orbital period (days)
t0	= ephemeris (BJD)
eps	= error tolerance for calculating the eccentric anomaly
t	= times (BJD) for which to calculate relative flux

Output:
mu_c 	= relative flux at times t
"""
def get_lc(e, aRs, i, u1, u2, p0, w, period, t0, eps,t):
        r_s = 1.0
        npoints = len(t)
        z0 = rsky(e, aRs, i, r_s, w, period, t0, eps, t)                #calculates separation of centers between the planet and the star
        mu_c = occultquad(z0, u1, u2, p0, npoints)   			#returns limb darkened model lightcurve
        return mu_c

e = 0.									#eccentricity		
aRs = 15.23								#semi-major axis to stellar radius ratio
i = 1.555								#inclination angle (radians)
u1 = 0.1								#quadratic limb darkening parameter 1
u2 = 0.3								#quadratic limb darkening paremeter 2
p0 = 0.115								#planet to star radius ratio
w = math.pi/2.								#argument of periapse
period = 1.58040464894        						#planet orbital period (days)
t0 = 2454966.52507           						#ephemeris (JD)
eps = 1.0e-7								#minimum eccentricity for solving Kepler's eqn (otherwise circular orbit assumed)
t = np.linspace(t0-period/20., t0 + period/20., 1000)

lc = get_lc(e, aRs, i, u1, u2, p0, w, period, t0, eps, t)

plt.plot(t, lc)
plt.xlabel("Time (days)")
plt.ylabel("Relative flux")
plt.ylim((0.984, 1.005))
plt.show()

