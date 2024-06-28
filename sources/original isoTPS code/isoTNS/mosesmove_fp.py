import numpy as np
import cPickle
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

#LOAD YOUR SPLITTER HERE  - see mine for conventions
#from MZsplitter import split_psi
from norm_splitter import split_psi as split_psi_n
from renyi_splitter import split_psi as split_psi_r


#Test-data Weak inter-chain 2-leg TFI ladder in paramagnetic phase
with open('test_data/2HAF_h-0.05_J2-0.5.mps', 'r') as f:
	Psi = cPickle.load(f)

### Convention for MPS to be split: Psi = [B0, B1, ...]
#      3
#      v
#      |
#  0->- -<-1
#      |
#      v
#      2
#
# 0, 1 are physical
# 2, 3 are virtual
# arrows denote the canonical form assumed
#
# Here B0 is the "bottom most" tensor (the first to be split), so is a wavefunction (all-in arrows)
# The test data is in this form.

### Pent is the 5-leg wavefunction
#      4
#	   |
#  2--- ---3
#     / \
#    0   1


### Tri is the 3-leg wavefunction formed by grouping [0 2], [4], [1 3] of Pent
#     1
#     |
# 0=== ===2
#
# 0 is the "special" leg to be split


###  Limits on split bond dimensions:
#
#          |
#        S |
#         / \
#  chi_V /   \  eta
#       /     \
#   ===/--->---====
#      a chi_H B


###  Leg ordering and arrows of returned A, S, B
#          mL
#          |
#        S |
#         / \
#        /   \
#       /     \
# d ---/----------- mR
#      A       B
#
#
#            1
#       A  _/
#          /`
#  0 ->---/--->- 2
#
#          1
#          |
#          v
#        S |
#        _/ \_
#        /` '\
#       0     2
#
#
#           1
#            _
#           '\  B
#             \
#         0->----<- 2
#

def moses_move(Psi, truncation_par = {'chi_max': {'eta_max':32, 'chiV_max':8, 'chiH_max':4}, 'p_trunc':1e-6 },):
	chi_max = truncation_par['chi_max']
	eta_max = chi_max['eta_max']
	chiV_max = chi_max['chiV_max']
	chiH_max = chi_max['chiH_max']
	
	print eta_max,chiV_max,chiH_max

	L = len(Psi)
	Lambda = []
	A = []

	#Current dimensions of the Pent tensor about to be split
	eta = 1
	chiV = 1
	pL = Psi[0].shape[0]
	pR = Psi[0].shape[1]
	chi = Psi[0].shape[3]
	
	#Initialize Pent from bottom of MPS
	Pent = Psi[0].reshape((chiV, eta, pL, pR, chi))
	for j in range(L):
		Tri = Pent.transpose([0, 2, 4, 1, 3])
		Tri = Tri.reshape((chiV*pL, chi, eta*pR))

		dL = np.min([chiV_max, Tri.shape[0]])
		dR = Tri.shape[0]/dL
		dR = np.min([dR, chiH_max])
		print "->",dL,dR,chiH_max
		
		if j==L-1:
			dL = 1
			dR = np.min([chiH_max*chiV_max, Tri.shape[0]])
		
		a, S, B, info = split_psi_r(Tri, dL, dR, truncation_par = {'chi_max':eta_max, 'p_trunc':truncation_par['p_trunc']} , verbose=0)
		a, S, B, info = split_psi_n(Tri, dL, dR, truncation_par = {'chi_max':eta_max, 'p_trunc':truncation_par['p_trunc']} , verbose=0,A=a)

		B = B.reshape((dR, B.shape[1], eta, pR)).transpose( [0, 3, 2, 1]) 
		a = a.reshape((chiV, pL, dL, dR)).transpose( [1, 3, 0, 2] )

		if j == L-1:
			B = B*S

		Lambda.append(B)
		A.append(a)
		
		if j < L-1:
			pL = Psi[j+1].shape[0]
			pR = Psi[j+1].shape[1]
			chi = Psi[j+1].shape[3]
			Pent = np.tensordot(S, Psi[j+1], axes = [[1], [2]])
			
		chiV = dL
		eta = B.shape[-1]
		
	return A, Lambda
	
def overlap(Psi,A,Lambda):
	L = len(Psi)
	O = np.ones([1,1])
	N = np.ones([1,1])
	for i in np.arange(L):
		d,d,chi1,chi2 = Psi[i].shape
		B = np.reshape(Psi[i],(d**2,chi1,chi2))
		C = np.tensordot(A[i],Lambda[i],axes=[1,0])
		C = np.transpose(C,(0,3,1,4,2,5))
		d,d,chiA_1,chiL_1,chiA_2,chiL_2 = C.shape
		C = np.reshape(C,(d*d,chiA_1*chiL_1,chiA_2*chiL_2))
		
		O = np.tensordot(O,np.conj(B), axes=(1,1))
		O = np.tensordot(O,C, axes=([0,1],[1,0]))
		O = np.transpose(O,(1,0))
		
		N = np.tensordot(N,np.conj(C), axes=(1,1))
		N = np.tensordot(N,C, axes=([0,1],[1,0]))
		N = np.transpose(N,(1,0))

	O = np.trace(O)
	N = np.trace(N)	
	return O/N

truncation_par = {'chi_max': {'eta_max':32, 'chiV_max':8, 'chiH_max':2},'p_trunc':1e-10 }
chiV_max_list = range(4,24,4)
err_list = []
for chiV_max in chiV_max_list:
	truncation_par['chi_max']['chiV_max']=chiV_max
	A, Lambda = moses_move(Psi,truncation_par=truncation_par)
	err_list.append(1-overlap(Psi,A,Lambda))
	print err_list[-1]
	exit()

plt.semilogy(chiV_max_list,err_list,'o-')
plt.ylabel('$\mathrm{error}$')
plt.xlabel('$\\chi_V$')
plt.xlim([0,35])
plt.savefig('out.pdf')
plt.show()
