import matplotlib.pyplot as pl
import numpy as np
#pl.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#params = {'text.usetex' : True,
#		'font.size' : 11,
#		'font.family' : 'lmodern',
#		'text.latex.unicode': True}
#pl.rcParams.update(params) 

J = 1
g = 3.1
e_ed = {4:-51.701896,5:-81.038243, 6:-116.9438125677, 8:-208.4631854858, 10:-326.260419}

cnt = 1
for L in [6,8,10]:
	pl.subplot(1,3,cnt)
	for f in [6,8]:
		leg = []
		for chi in [2,4,6]:
			eta = f*chi
			leg.append("$\\chi = %.0f$"%chi)
			E = np.loadtxt("ising_moses_J_%.02f"%J + "_g_%.02f"%g+ "_L_%.0f"%L + "_eta_%.0f"%eta + "_chi_%.0f"%chi)
			pl.loglog(E[0],(E[1]-e_ed[L])/np.abs(e_ed[L]),'o-')
			ax = pl.gca()
			if cnt>1:
				pl.yticks([])
			else:
				pl.ylabel("$\\Delta E$")
			pl.xlabel("$d\\tau$")
			pl.title("$L=%.0f$"%L)
		ax.set_prop_cycle(None)
		print leg
		if L==6:
			pl.legend(leg)
	cnt = cnt + 1
	pl.ylim([7e-5,1e-1])
pl.savefig('TEBD.pdf')
pl.show()
