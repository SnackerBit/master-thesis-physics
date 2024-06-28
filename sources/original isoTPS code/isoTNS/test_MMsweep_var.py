from mosesmove import moses_move, sweeped_moses_move, check_overlap
from misc import *
import cPickle 
from matplotlib import pyplot as plt
from test_iter import var_moses
from var_splitter import var_moses

"""This code takes an n-leg TFI ladder, and decomposes it as

	Psi = A0 A1 A2 Lambda3
	
	weeeeeee
"""

np.set_printoptions(precision = 4, suppress = True, linewidth=120)


def S(s):
	return -2*np.vdot(s**2, np.log(s))


def split_columns(Psi, n, truncation_par):
	""" Given an MPO Psi for an n-leg ladder, with physical legs shaped as d, d^(n-1), succesively perform the tri-split Moses Move to bring into a canonical PEPs. Returns the isometries "As" and the renormalized set of Lambdas.

 """

	print "Initial chi",  np.max( [b.shape[3] for b in Psi])
	
	d = Psi[0].shape[0] 
	As = []
	Ls = [Psi]
	Er = []

	for j in range(n-1):
		#A, Lambda, info = sweeped_moses_move(Psi, truncation_par)
		A, Lambda, info = moses_move(Psi, truncation_par)#, schedule = [4])
		A, Lambda = var_moses(Psi,A,Lambda,truncation_par['chi_max']['etaV_max'],N = 10)
		As.append(A)
		Ls.append(Lambda)
		Er.append(check_overlap(Psi, A, Lambda))
		print "--"
		print check_overlap(Psi, A, Lambda)
		print "--"
		Psi = peel(Lambda, d)
		
		print  "|Psi - A Lam|^2 / L :", Er[-1]/len(Psi)
		print "Errors", info['errors']
		print "A.chiV", [a.shape[3] for a in A]
		print "A.chiH", [a.shape[1] for a in A]
		print "L.chiV", [l.shape[3] for l in Lambda]
		print

	return As, Ls, Er


if __name__ == '__main__':
	with open('test_data/3TFI_JH0.5.mps', 'r') as f:
		Psi = cPickle.load(f)
	n = 3

	truncation_par = {'chi_max': {'eta0_max':16, 'eta1_max': 32, 'chiV_max':4, 'chiH_max':4}, 'p_trunc':1e-8 }
	As, Ls, Er = split_columns(Psi, n, truncation_par)

	plt.gcf().clear()
	plt.subplot(211)
	Ss = []
	for psi in Ls:
		s = mps_entanglement_spectrum(psi)
		Ss.append(s)
		plt.plot([S(p) for p in s])

	plt.ylabel(r'$S_V(x)$', fontsize=16)
	plt.xlabel(r'$x$', fontsize=16)
	plt.legend(range(n))
	
	plt.subplot(212)
	plt.plot(Ss[0][10], '.-')
	plt.plot(Ss[1][10], '.-')
	plt.plot(Ss[2][10], '.-')
	plt.plot(Ss[3][10], '.-')
	plt.yscale('log')
	plt.legend([0, 1])
	plt.show()
	exit()

	plt.figure()

	for e in Er:
		print e
		plt.semilogy(e)
	plt.show()
