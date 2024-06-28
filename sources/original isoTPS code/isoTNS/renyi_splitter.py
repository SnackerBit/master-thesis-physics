import numpy as np
import cPickle
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

### psi is the 3-leg wavefunction
#     1
#     |
# 0--- ---2
#    psi
#
# 0 is the "special" leg to be split


###  Limits on split bond dimensions:
#          mL
#          |
#        S |
#         / \
#    dL  /   \  chi_max
#       /     \
# d ---/----------- mR
#      A  dR   B
#

### Leg ordering and arrows of returned A, S, B
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

def split_psi(Psi, dL, dR, truncation_par = {'chi_max':32, 'p_trunc':1e-6}, verbose=0, max_iter=50):
	chi = 1
	trunc_leg = 1
	d,mL,mR = Psi.shape
	
	A,Psi_p,trunc_leg = disentangle_split_Psi_helper(Psi, dL ,dR ,truncation_par['chi_max'], max_iter,verbose=0)

	Psi_p = np.transpose(Psi_p,[1,0,2,3]) # dL mL dR mR
	Psi_p = np.reshape(Psi_p,(dL*mL,dR*mR))

	X,s2,Z,chi_c,trunc_bond = svd_trunc(Psi_p,truncation_par['chi_max'] ,remove_noise = True)
	S = np.reshape(X,(dL,mL,chi_c))
	S = np.tensordot(S,np.diag(s2),axes=(2,0))
	
	B = np.reshape(Z,(chi_c,dR,mR))
	B = np.transpose(B,(1,0,2))
	
	info = {'error':trunc_leg}
	return A, S, B, info
	

def disentangle_split_Psi_helper(Psi, dL, dR, m, max_iter, verbose=0):
	""" Solve Psi = A.Lambda where A is an isometry that "splits" a leg using disentangle_Psi
	
		Psi has a physical leg (d) and L/R auxilliaries (mL, mR)
		
		A:d ---> dL x dR is the isometric splitting; |dL x dR| < d (isometry)
		
		Psi:  d, mL, mR
		A: d, dL, dR
		Lambda: mL,dL, dR, mR
		
		return A, Lambda,trunc_leg
	"""
	# Get the isometry
	d,mL,mR = Psi.shape
	theta = np.reshape(Psi,(d,mL*mR)).copy()
	X,y,Z,D2,trunc_leg = svd_trunc(theta,dL*dR,remove_noise = False)
	A = X
	if D2 < dL*dR:
		dL = int(np.sqrt(D2))
		dR = int(np.sqrt(D2))
	
	# Disentangle the two-site wavefunction
	theta = np.tensordot(np.diag(y),Z,axes=([1],[0]))
	theta = np.reshape(theta,(dL,dR,mL,mR))
	theta = np.transpose(theta,(2,0,1,3))
	thetap,U,S = disentangle_Psi(theta, eps = 1e-12,  n = 2, max_iter = max_iter, verbose = verbose)

	A = np.tensordot(A,np.reshape(np.conj(U),(dL,dR,dL*dR)),axes=([1],[2]))
	return A,thetap,trunc_leg

def U2(psi):
	"""Entanglement minimization via 2nd Renyi entropy
	
		Returns S2 and 2-site U
	"""
	chi = psi.shape
	rhoL = np.tensordot(psi, np.conj(psi), axes = [[2, 3], [2, 3]])
	dS = np.tensordot(rhoL, psi, axes = [[2, 3], [0, 1] ])
	dS = np.tensordot( np.conj(psi), dS, axes = [[0, 3], [0, 3]])
	dS = dS.reshape((chi[1]*chi[2], -1))
	s2 = np.trace( dS )

	X, Y, Z = np.linalg.svd(dS)
	#print (np.dot(X, Z).T).conj()
	return -np.log(s2), (np.dot(X, Z).T).conj()

def disentangle_Psi(psi, eps = 1e-12,  n = 2, max_iter = 30, verbose = 0):
	""" Disentangles a 2-site TEBD-style wavefunction.
	
		psi = mL, dL, dR, mR		; mL, mR are like "auxilliary" and dL, dR the "physical"
	
		Find the 2-site unitary U,
		
			psi --> U.psi = psi', U = dL, dR, dL, dR
	
		which minimizes the nth L/R Renyi entropy.
		
		Returns psi', U, Ss, where Ss are entropies during iterations
	
	"""

	if n!=2:
		raise NotImplemented
	
	Ss = []
	chi = psi.shape
	U = np.eye(chi[1]*chi[2], dtype = psi.dtype)
	m = 0
	go = True
	while m < max_iter and go:
		s, u = U2(psi) #find 2-site unitary
		U = np.dot(u, U)
		u = u.reshape( (chi[1], chi[2], chi[1], chi[2]))
		psi = np.tensordot(u, psi, axes = [[2, 3], [1, 2]]).transpose([2, 0, 1, 3])
		Ss.append(s)
		if m > 1:
			go = Ss[-2] - Ss[-1] > eps
		m+=1

	if verbose:
		print "Evaluations:", m, "dS", -np.diff(Ss)
	if verbose > 1:
		plt.subplot(1, 3, 2)
		plt.plot(Ss, '.-')
		plt.subplot(1, 3, 3)
		plt.plot(-np.diff(Ss), '.-')
		plt.yscale('log')
		plt.tight_layout()
		plt.show()
	return psi, U.reshape([chi[1], chi[2], chi[1], chi[2]]), Ss
	
def svd_trunc(A,chi_max,remove_noise = False):
	""" SVD of A
		
		A: numpy matrix
		chi_max : keeps at most chi_max singular values
		remove_noise : If true, small singular values are removed
		
		returns: X,y,Z,chi_final,trunc 
	"""
	
	try:
		X, y, Z = np.linalg.svd(A,full_matrices=0)			
	except np.linalg.LinAlgError:
		print "Using DGESVD!"
		X, y, Z = svd(A,full_matrices=0)			
		
	if remove_noise:
		chi_final = np.min([np.sum(y>10.**(-12)), chi_max])	
	else:
		chi_final = np.min([len(y), chi_max])
		
	norm = np.linalg.norm(y[:chi_final])
	trunc = np.sum(y[chi_final:]**2)
	y = y[0:chi_final]/norm
		
	X = X[:,0:chi_final] 
	Z = Z[0:chi_final,:]

	return X,y,Z,chi_final,trunc

