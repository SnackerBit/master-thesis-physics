import matplotlib.pyplot as pl
import numpy as np
#pl.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#params = {'text.usetex' : True,
#		'font.size' : 11,
#		'font.family' : 'lmodern',
#		'text.latex.unicode': True}
#pl.rcParams.update(params) 

J = 1
g = 3.5
e_ed = {4:-57.82436977,5:-90.5608793455, 6:-130.6117109064, 8: -232.656314635, 10:-363.95737940}
pl.figure(figsize=(6,3.2))
cnt = 1
for L in [6,8,10]:
	pl.subplot(1,3,cnt)
	for f in [6]:
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
		print(leg)
		if L==8:
			pl.legend(leg)
	cnt = cnt + 1
	pl.ylim([3e-6,5e-2])
	pl.xlim([0.01,0.5])
pl.plot(E[0],E[0]**2/30.,'--k')
pl.plot(E[0],1./E[0]/100000.,'--k')

pl.tight_layout()
pl.savefig('TEBD2_3.5.pdf')
pl.show()
