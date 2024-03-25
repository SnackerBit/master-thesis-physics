
import matplotlib.pyplot as pl
import numpy as np

J = 1
g = 3.04
f = 5
L = 5

for chi in range(4,16,2):
	eta = f*chi
	E = np.loadtxt("ising_moses_J_%.02f"%J + "_g_%.02f"%g+ "_L_%.0f"%L + "_eta_%.0f"%eta + "_chi_%.0f"%chi)
	pl.loglog(E[0],E[1],'o-')
pl.ylim([1e-6,1e-1])
pl.show()
