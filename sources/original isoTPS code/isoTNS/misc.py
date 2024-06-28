import numpy as np
import scipy as sp
from scipy import linalg
import svd_dgesvd
from itertools import izip

def group_legs(a, axes):
	""" Given list of lists like axes = [ [l1, l2], [l3], [l4 . . . ]]
	
		does a transposition of "a" according to l1 l2 l3... followed by a reshape according to parantheses.
		
		Return the reformed tensor along with a "pipe" which can be used to undo the move
	"""

	nums = [len(k) for k in axes]


	flat = []
	for ax in axes:
		flat.extend(ax)

	a = np.transpose(a, flat)
	perm = np.argsort(flat)

	oldshape = a.shape

	shape = []
	oldshape = []
	m = 0
	for n in nums:
		shape.append(np.prod(a.shape[m:m+n]))
		oldshape.append(a.shape[m:m+n])
		m+=n
	
	a = np.reshape(a, shape)
	
	pipe = (oldshape, perm)

	return a, pipe

def ungroup_legs(a, pipe):
	"""
		Given the output of group_legs,  recovers the original tensor (inverse operation)
		
		For any singleton grouping [l],  allows the dimension to have changed (the new dim is inferred from 'a').
	"""
	if a.ndim!=len(pipe[0]):
		raise ValueError
	shape = []
	for j in range(a.ndim):
		if len(pipe[0][j])==1:
			shape.append(a.shape[j])
		else:
			shape.extend(pipe[0][j])

	a = a.reshape(shape)
	a = a.transpose(pipe[1])
	return a

def transpose_mpo(Psi):
	"""Transpose row / column of an MPO"""
	return [ b.transpose([1, 0, 2, 3]) for b in Psi]

def mps_group_legs(Psi, axes = 'all'):
	""" Given an 'MPS' with a higher number of physical legs (say, 2 or 3), with B tensors

			physical leg_1 x physical leg_2 x . . . x virtual_left x virtual_right
			
		groups the physical legs according to axes = [ [l1, l2], [l3], . .. ] etc, 
		
		Example:
		
		
			Psi-rank 2,	axes = [[0, 1]]  will take MPO--> MPS
			Psi-rank 2, axes = [[1], [0]] will transpose MPO
			Psi-rank 3, axes = [[0], [1, 2]] will take to MPO
		
		If axes = 'all', groups all of them together.
		
		Returns:
			Psi
			pipes: list which will undo operation
	"""
	
	if axes == 'all':
		axes = [ range(Psi[0].ndim-2) ]
	
	psi = []
	pipes = []
	for j in range(len(Psi)):
		ndim = Psi[j].ndim
		b, pipe = group_legs( Psi[j], axes + [[ndim-2], [ndim-1]])
		
		psi.append(b)
		pipes.append(pipe)

	return psi, pipes

def mps_ungroup_legs(Psi, pipes):
	"""Inverts mps_group_legs given its output"""
	psi = []
	for j in range(len(Psi)):
		psi.append(ungroup_legs( Psi[j], pipes[j]))
	
	return psi



def mps_invert(Psi):
	np = Psi[0].ndim - 2
	return [ b.transpose(range(np) + [-1, -2]) for b in Psi[::-1] ]

def mps_2form(Psi, form = 'A'):
	"""Puts an mps with an arbitrary # of legs into A or B-canonical form
		
		hahaha so clever!!!
	"""
	Psi, pipes = mps_group_legs(Psi, axes='all')

	if form=='B':
		Psi = [ b.transpose([0, 2, 1]) for b in Psi[::-1] ]
	
	L = len(Psi)
	T = Psi[0]
	for j in range(L-1):
		T, pipe = group_legs(T, [[0, 1], [2]]) #view as matrix
		A, s = np.linalg.qr(T) #T = A s can be given from QR
		Psi[j] = ungroup_legs(A, pipe)
		T = np.tensordot( s, Psi[j+1], axes = [[1], [1]]).transpose([1, 0, 2]) #Absorb s into next tensor

	Psi[L-1] = T

	if form=='B':
		Psi = [ b.transpose([0, 2, 1]) for b in Psi[::-1] ]
	
	Psi =  mps_ungroup_legs(Psi, pipes)

	return Psi

def mps_entanglement_spectrum(Psi):
	
	Psi, pipes = mps_group_legs(Psi, axes='all')


	#First bring to A-form
	L = len(Psi)
	T = Psi[0]
	for j in range(L-1):
		T, pipe = group_legs(T, [[0, 1], [2]]) #view as matrix
		A, s = np.linalg.qr(T) #T = A s can be given from QR
		Psi[j] = ungroup_legs(A, pipe)
		T = np.tensordot( s, Psi[j+1], axes = [[1], [1]]).transpose([1, 0, 2]) #Absorb s into next tensor

	Psi[L-1] = T

	#Flip the MPS around
	Psi = [ b.transpose([0, 2, 1]) for b in Psi[::-1] ]

	T = Psi[0]
	Ss = []
	for j in range(L-1):
		T, pipe = group_legs(T, [[0, 1], [2]]) #view as matrix
		U, s, V = np.linalg.svd(T) #T = A s can be given from QR
		Ss.append(s)
		Psi[j] = ungroup_legs(U, pipe)
		s = ((V.T)*s).T
		T = np.tensordot( s, Psi[j+1], axes = [[1], [1]]).transpose([1, 0, 2]) #Absorb sV into next tensor

	return Ss


def mpo_on_mpo(X, Y, form = None):
	""" Multiplies two two-sided MPS, XY = X*Y and optionally puts in a canonical form
	"""
	if X[0].ndim!=4 or Y[0].ndim!=4:
		raise ValueError
	
	XY = [  group_legs( np.tensordot(x, y, axes = [[1], [0]]), [[0], [3], [1, 4], [2, 5]] )[0] for x, y in izip(X, Y)]
	
	if form is not None:
		XY = mps_2form(XY, form)
	
	return XY

def svd(theta, compute_uv=True, full_matrices=True):
	try:
		if compute_uv:
			U, s, V = np.linalg.svd( theta, compute_uv=compute_uv, full_matrices=full_matrices)
		else:
			s = np.linalg.svd( theta, compute_uv=compute_uv, full_matrices=full_matrices)
	
	except np.linalg.linalg.LinAlgError:
		print "*dgesvd*"
		if compute_uv:
			U, s, V = svd_dgesvd.svd_dgesvd(theta, full_matrices = full_matrices, compute_uv = compute_uv)
		else:
			s = svd( theta, compute_uv=compute_uv, full_matrices=full_matrices)
	
	if compute_uv:	
		return U,s,V
	else:
		return s	

def svd_theta(theta, truncation_par):
	""" SVD and truncate a matrix based on truncation_par """
	
	try:
		U, s, V = svd( theta, compute_uv=True, full_matrices=False)
	
	except np.linalg.linalg.LinAlgError:
		print "*dgesvd*"
		U, s, V = svd_dgesvd.svd_dgesvd(theta, full_matrices = 0, compute_uv = 1)
	
	nrm = np.linalg.norm(s)
	eta = np.min([ np.count_nonzero((1 - np.cumsum(s**2)/nrm**2) > truncation_par['p_trunc'])+1, truncation_par['chi_max']])
	nrm_t = np.linalg.norm(s[:eta])
	#print "s", s
	A = U[:, :eta]
	SB = ((V[:eta, :].T)*s[:eta]/nrm_t).T
	
	info = {'p_trunc': 1 - (nrm_t/nrm)**2, 's': s[:eta], 'nrm':nrm}
	return A, SB, info




