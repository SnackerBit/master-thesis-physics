from misc import *
from mosesmove import moses_move
import sys
from mosesmove import *

### PEPs      4
#             |
#             |
#       1 --------- 2
#            /|
#     t =   / |
#          0  3
#
#
#	Stored in cartesian form PEPs[x][y], list-of-lists


def tebd_on_mps(Psi, U, truncation_par, order = 'R', reduced_update = True, O = None):
	"""
		Applies Trotter decomposed circuit
		
			U = U0 U1 . . . Un
			
		on an MPS. If order ='R', sweeps from left-to-right, assuming Psi starts in B-form, and will end in A-form. order = 'L' gives the opposite.
		
		For a 2nd-order algorithm, call
		
			Psi = tebd_on_mps(tebd_on_mps(Psi, U, 'R'), U, 'L')
		
		which takes B-form to B-form.
		
		Psi is allowed to have more than one physical leg. The gate U  is assumed to act ONLY on the first leg (this is  useful for our PEPs case).
		
		If reduced_update = True, follows Fig. 4 of https://arxiv.org/pdf/1503.05345v2.pdf using a QR decomposition
		
	"""

	L = len(Psi)

	def sweep(Psi, U, O = None):
		""" Left to right sweep """
		psi = Psi[0]
		num_p = psi.ndim - 2 #number of physical legs
		p_trunc = 0.
		nrm = 1.
		
		expectation_O = []
		
		for j in range(L-1):


			psi, pipe1L = group_legs(psi, [ [0], range(1,num_p+1), [num_p + 1] ]  )   # bring to 3-leg form
			B, pipe1R = group_legs(Psi[j+1], [ [0], [num_p], range(1,num_p) + [num_p + 1] ] ) # bring to 3-leg form
			#if np.abs(np.linalg.norm(psi) - 1) > 1e-10:
			#	print "|psi|_", j, np.linalg.norm(psi)
			if reduced_update and psi.shape[0]*psi.shape[2] < psi.shape[1]:
				reduced_L = True
				psi, pipe2L = group_legs(psi, [ [1], [0, 2] ] )
				QL, psi = np.linalg.qr(psi)
				psi = ungroup_legs(psi, pipe2L)
			else:
				reduced_L = False
			
			if reduced_update and B.shape[0]*B.shape[1] < B.shape[2]:
				reduced_R = True
				B, pipe2R = group_legs(B, [[0, 1], [2] ] )
				QR, B = np.linalg.qr(B.T)
				QR = QR.T; B = B.T
				B = ungroup_legs(B, pipe2R)
			else:
				reduced_R = False
			#if np.abs(np.linalg.norm(psi) - 1) > 1e-10:
			#	print "|psi| in tebd before U", np.linalg.norm(psi),
			theta = np.tensordot( psi, B, axes = [[-1], [-2]]) #Theta = s B
			#if np.abs(np.linalg.norm(theta) - 1) > 1e-10:
			#	print "|theta| in tebd before U", np.linalg.norm(theta)
			if O is not None:
				Otheta = np.tensordot( O[j%len(O)], theta, axes = [[2, 3], [0, 2] ])
				expectation_O.append( np.tensordot(theta.conj(), Otheta, axes = [[0, 2, 1, 3], [0, 1, 2, 3]] ))
				
			if U is not None:
				theta = np.tensordot( U[j%len(U)], theta, axes = [[2, 3], [0, 2] ]) #Theta = U Theta
			else:
				theta = theta.transpose([0, 2, 1, 3])
			
			theta, pipeT = group_legs( theta, [[0, 2], [1, 3]]) #Turn into a matrix
			
			A, SB, info = svd_theta(theta, truncation_par) #Theta = A s
			
			#Back to 3-leg form
			A = A.reshape( pipeT[0][0] + (-1, ))
			SB = SB.reshape( (-1, ) + pipeT[0][1]).transpose([1, 0, 2])
			
			if reduced_L:
				A = np.tensordot(QL, A, axes = [[1], [1]]).transpose([1, 0, 2])
			if reduced_R:
				SB = np.dot(SB, QR)
			
			A = ungroup_legs(A, pipe1L)
			SB = ungroup_legs(SB, pipe1R)
			
			p_trunc += info['p_trunc']
			nrm*= info['nrm']
			
			Psi[j] = A
			psi = SB
		
		Psi[L-1] = psi
		
		return p_trunc, nrm, expectation_O

	if order == 'R':
		p_trunc, nrm, expectation_O = sweep(Psi, U, O)
	
	if order == 'L':
		Psi = mps_invert(Psi)
		if U is not None:
			U = [ u.transpose([1, 0, 3, 2]) for u in U[::-1] ]
		if O is not None:
			O = [ o.transpose([1, 0, 3, 2]) for o in O[::-1] ]
		p_trunc, nrm, expectation_O = sweep(Psi, U, O )
		expectation_O = expectation_O[::-1]
		Psi = mps_invert(Psi)

	info = {'p_trunc':p_trunc, 'nrm':nrm, 'expectation_O':expectation_O}
	
	return Psi, info

def peps_sweep(PEPs, U, truncation_par, O = [None], verbose = 0):
	""" Given a PEPs in (-+) form, with wavefunction in upper-left corner, sweeps right to bring to (++) form, with wavefunction in upper-right corner.
	
		Along the way, applies  2-site TEBD-gates U[j] on column[j] and measures  2-site ops O[j]
		
		
	"""

	Psi = PEPs[0] #One-column wavefunction
	Lx = len(PEPs)
	Ly = len(PEPs[0])
	min_p_trunc = truncation_par['p_trunc']
	target_p_trunc = 0.01*truncation_par['p_trunc'] #This will be adjusted according to moses-truncation errors
	
	nrm = 1.
	tebd_error  = []
	moses_error = []
	moses_d_error = []
	eta0 = np.ones((Lx - 1, Ly), dtype = np.int)
	eta1 = np.ones((Lx, Ly), dtype = np.int)
	expectation_O = []
	
	
	if U is None:
		U = [None]
	if O is None:
		O = [None]
	
	for j in range(Lx):
		#Psi is in A-form
		
		Psi, info = tebd_on_mps(Psi, U[j%len(U)], truncation_par = {'p_trunc':target_p_trunc, 'chi_max': truncation_par['chi_max']['eta1_max']}, order = 'L', O = O[j%len(O)]) #TEBD sweeps DOWN, putting it in B-form
		
		tebd_error.append(info['p_trunc'])
		nrm*=info['nrm']
		expectation_O.append(info['expectation_O'])
		eta1[j][:] = [l.shape[4] for l in Psi]
		#Psi is now in B-form
		
		if j < Lx - 1:
			Psi, pipe = mps_group_legs(Psi, [[0, 1], [2]]) #View as MPO
			A, Lambda, info = moses_move(Psi, truncation_par, verbose = verbose) #Moses move puts it back in A-form
			#A, Lambda, info = sweeped_moses_move(Psi, truncation_par, verbose = verbose)
			moses_error.append(info['total_error'])
			moses_d_error.append(info['total_d_error'])
			target_p_trunc = np.max([0.1*info['total_error']/len(Psi), min_p_trunc])
			A = mps_ungroup_legs(A, pipe)
			PEPs[j] = A
			Psi, pipe = mps_group_legs(PEPs[j+1], axes = [ [1], [0, 2] ] ) #Tack Lambda onto next column: Psi = Lambda B
			Psi = mpo_on_mpo(Lambda, Psi)
			Psi = mps_ungroup_legs(Psi, pipe)
			PEPs[j+1] = Psi
			eta0[j][:] = [l.shape[3] for l in Lambda]
		else:
			Psi = mps_2form(Psi, 'A') #TODO perhaps mps_2form should have a truncation-parameter to drop small s
			PEPs[j] = Psi

	if verbose:
		print ("{:>8.1e} "*Lx).format(*tebd_error)
		print "    ",
		print ("{:>8.1e} "*(Lx-1)).format(*moses_error)
		print "    ",
		print ("{:>8.1e} "*(Lx-1)).format(*moses_d_error)
		print

	return PEPs, {'nrm':nrm, 'expectation_O': expectation_O, 'moses_error':moses_error,'moses_d_error':moses_d_error, 'tebd_error':tebd_error, 'eta0':eta0, 'eta1':eta1}

def peps_ESWN_tebd(PEPs, Us, truncation_par, Os = None, verbose = 0):
	""" Applies four sweeps of TEBD to a PEPs as follows:
	
	  
	  		1) Starting in B = (-+) form,  sweep right (east) applying a set of 1-column gates Us[0] , bringing to A = (++) form
			2) Rotate network 90-degrees counterclockwise, so that effectively (++) ---> (-+)
			
			repeat 4x
		
		Us = [UE, US, UW, UN] ; each Ui[x][y] is 2-site gate on column x, bond y when sweep is moving in direction 'i'.
		
		
		I am assuming some sort of inversion symmetry in the gates UW, later will have to fix a convention.
	
	"""
	
	def rotT(T):
		""" 90-degree counter clockwise rotation of tensor """
		return np.transpose(T, [0, 4, 3, 1, 2])

	def peps_rotate(PEPs):
		""" 90-degree counter clockwise rotation of PEPs """
		Lx = len(PEPs)
		Ly = len(PEPs[0])

		rPEPs = [ [None]*Lx for y in range(Ly)]
		for y in range(Ly):
			for x in range(Lx):
				#print  y, x, "<---", x, Ly - y - 1, rotT(PEPs[x][Ly - y - 1]).shape
				rPEPs[y][x] = rotT(PEPs[x][Ly - y - 1]).copy()

		return rPEPs
	
	nrm = 1.
	moses_d_error = 0.
	moses_error = 0.
	tebd_error = 0.
	eta0_avg = 0.
	eta1_avg = 0.
	expectation_O = []
	if Us is None:
		Us = [[None]]*4
	if Os is None:
		Os  = [[None]]*4
	
	for j in range(4):

		PEPs, info = peps_sweep(PEPs, Us[j], truncation_par, Os[j], verbose = 0) #TODO . . . some possible inversion action on Us? Depends on convention
		
		expectation_O.append(info['expectation_O'])
		nrm*=info['nrm']
		moses_d_error+=np.sum(info['moses_d_error'])/4. #Total error / sweep
		moses_error+=np.sum(info['moses_error'])/4. #Total error / sweep
		tebd_error+=np.sum(info['tebd_error'])/4.
		#eta0_avg+=np.sum( np.log(info['eta0']).flat)/(4*len(info['eta0'].flat))
		#eta1_avg+=np.sum( np.log(info['eta1']).flat)/(4*len(info['eta1'].flat))
		eta0_avg+=np.sum( (info['eta0']**2).flat)/(4*len(info['eta0'].flat))
		eta1_avg+=np.sum( (info['eta1']**2).flat)/(4*len(info['eta1'].flat))
		
		PEPs = peps_rotate(PEPs)
	
	#eta0_avg = np.exp(eta0_avg)
	#eta1_avg = np.exp(eta1_avg)
	eta0_avg = np.sqrt(eta0_avg)
	eta1_avg = np.sqrt(eta1_avg)
	return PEPs, {'nrm':nrm, 'moses_error':moses_error, 'moses_d_error':moses_d_error, 'tebd_error':tebd_error, 'eta0_max':np.max(info['eta0'].flat), 'eta1_max':np.max(info['eta1'].flat), 'eta0_avg':eta0_avg, 'eta1_avg':eta1_avg, 'expectation_O':expectation_O}


def peps_EW_tebd(PEPs, Us, truncation_par, Os = None, verbose = 1):
	""" Applies two sweeps of TEBD to a PEPs, effectively E moving, then W moving, as follows:
	
	  
	  		1) Starting in B = (-+) form,  sweep right (east) applying a set of 1-column gates Us[0] , bringing to A = (++) form
			2) Reflect network according to x <-->-x,  taking (++) ---> (-+)
			3) Sweep East again
			4) Reflect again.
			
		effectively implementing an E-W sweep.
		
		
		Us = [UN, UN] ; each Ui[x][y] is 2-site gate on column x, bond y.
	"""
	
	if Us is None:
		Us = [[None]]*2
	if Os is None:
		Os  = [[None]]*2

	def refT(T):
		"""   x < ----> -x reflection of tensor """
		return np.transpose(T, [0, 2, 1, 3, 4])

	def peps_reflect(PEPs):
		"""  x < ----> -x reflection of PEPs """
		Lx = len(PEPs)
		Ly = len(PEPs[0])

		rPEPs = [ [None]*Ly for x in range(Lx)]
		for x in range(Lx):
			for y in range(Ly):
				rPEPs[x][y] = refT(PEPs[Lx - x - 1][y]).copy()

		return rPEPs
	
	nrm = 1.
	moses_error = 0.
	tebd_error = 0.
	eta0_avg = 0.
	eta1_avg = 0.
	expectation_O = []
	
	for j in range(2):

		PEPs, info = peps_sweep(PEPs, Us[j], truncation_par) #TODO . . . some possible inversion action on Us? Depends on convention
		
		expectation_O.append(info['expectation_O'])
		nrm*=info['nrm']
		moses_error+=np.sum(info['moses_error'])/2. #Total error / sweep
		tebd_error+=np.sum(info['tebd_error'])/2.
		#eta0_avg+=np.sum( np.log(info['eta0']).flat)/(4*len(info['eta0'].flat))
		#eta1_avg+=np.sum( np.log(info['eta1']).flat)/(4*len(info['eta1'].flat))
		eta0_avg+=np.sum( (info['eta0']**2).flat)/(2*len(info['eta0'].flat))
		eta1_avg+=np.sum( (info['eta1']**2).flat)/(2*len(info['eta1'].flat))
		
		PEPs = peps_reflect(PEPs)
	
	#eta0_avg = np.exp(eta0_avg)
	#eta1_avg = np.exp(eta1_avg)
	eta0_avg = np.sqrt(eta0_avg)
	eta1_avg = np.sqrt(eta1_avg)
	return PEPs, {'nrm':nrm, 'moses_error':moses_error, 'tebd_error':tebd_error, 'eta0_max':np.max(info['eta0'].flat), 'eta1_max':np.max(info['eta1'].flat), 'eta0_avg':eta0_avg, 'eta1_avg':eta1_avg, 'expectation_O':expectation_O}


def peps_print_chi(PEPs):
	"""	o-- cH--o
		|       |
	    cV      |
		|       |
		o--xxx--o """

	Lx = len(PEPs)
	Ly = len(PEPs[0])
	for y in range(Ly-1, -1, -1):
	
		print (" --{:^3d}--"*(Lx-1)).format(*[t.shape[2] for t in  [ p[y] for p in PEPs[:-1]]  ]  ) #+ "X"*(y==(Ly-1))
		if y > 0:
			print ("|       "*(Lx))
			print ("{:<3d}     "*(Lx)).format(*[t.shape[3] for t in [ p[y] for p in PEPs] ] )
			print ("|       "*(Lx))


def peps_check_sanity(PEPs):
	""" Just checks compatible bond dimensions"""
	Lx = len(PEPs)
	Ly = len(PEPs[0])

	for x in range(Lx):
		for y in range(Ly):
			if x < Lx - 1:
				
				assert PEPs[x][y].shape[2]==PEPs[x+1][y].shape[1], "{} {} {}".format(x, y, PEPs[x][y].shape)

			if y < Ly - 1:
				assert PEPs[x][y].shape[4]==PEPs[x][y+1].shape[3], "{} {}".format(x, y)

			assert PEPs[x][y].shape[0]==2


def H_TFI(L, g, J=1):
	""" List of gates for TFI = -g X - J ZZ. Deals with edges to make gX uniform everywhere """


	sx = np.array([[0,1],[1,0]])
	sz = np.array([[1,0],[0,-1]])
	id = np.eye(2)
	d = 2

	def h(gl, gr, J):

		return (-np.kron(sz, sz)*J - gr*np.kron(id, sx) - gl*np.kron(sx, id)).reshape([d]*4)
	
	H = []
	for j in range(L-1):
		if j==0:
			gl = g
		else:
			gl = 0.5*g
		if j==L-2:
			gr = 1.*g
		else:
			gr = 0.5*g
		H.append(h(gl, gr, J))

	return [H]

def H_XXZ(L, Delta, h = 0., J = 1.):
	""" List of gates for XXZ =  XX + YY + Delta ZZ + h (-1)^(x+y) Z
	
		Does change of basis of "X" on every-other site, so
		
						XXZ' =  XX - YY - Delta ZZ - h Z
	
	"""
	sx = np.array([[0,1],[1,0]])/2.
	sz = np.array([[1,0],[0,-1]])/2.
	sy = np.array([[0,-1j],[1j,0.]])/2.
	id = np.eye(2)
	d = 2

	def hi(J, hl, hr):
		return (J*(-np.kron(sz, sz)*Delta + np.real( np.kron(sx, sx) - np.kron(sy, sy))) - hl*np.kron(sz, id) - hr*np.kron(id, sz)).reshape([d]*4)

	H = []
	for j in range(L-1):
		if j==0:
			hl = h
		else:
			hl = 0.5*h
		
		if j==L-2:
			hr = 1.*h
		else:
			hr = 0.5*h
		H.append(hi(J, hl, hr))

	return [H]

def make_U(H, t):
	""" U = exp(-t H) """
	d = H[0][0].shape[0]
	return [[sp.linalg.expm(-t*h.reshape( (d**2, -1) )).reshape([d]*4) for h in Hc] for Hc in H]

def p2_tebd(PEPs, H, dt, N, truncation_par, verbose = 0):
	""" Nsteps of 2nd-order Trotter

		Hv/2  Hh Hv Hh . . . Hh Hv/2
	
		given Hv, Hh = H .
		
	"""
	E_nrm = T = 0.
	p_trunc = truncation_par['p_trunc']
	
	V = len(PEPs)*len(PEPs[0])
	
	Uh = make_U(H[0], dt)
	Uv = make_U(H[1], dt)
	Uv2 = make_U(H[1], dt/2.)  #HALF STEP (for 2nd order trotter)
	Hv2 = [ [h/2. for h in Hc] for Hc in H[1]]
	
	PEPs, info = peps_ESWN_tebd(PEPs, [Uv2, Uh, Uv, Uh], truncation_par, Os = None)
	m_error = info['moses_error']/V
	m_d_error = info['moses_d_error']/V
	tebd_error = info['tebd_error']/V
	H_exp = np.sum(info['expectation_O'])/V
	H_nrm = -np.log(info['nrm'])/V/dt
	if verbose:
		print ("{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}").format('Energy(nrm)', '<H>', '<eta0>', 'max(eta0)', '<eta1>', 'max(eta1)', 'moses_d_err', 'moses_err', 'tebd_err')
		print ("{:>12.7f}{:>12.7f}{:>12.2f}{:>12.2f}{:>12.2f}{:>12.2f}{:>12.2e}{:>12.2e}{:>12.2e}").format(np.nan,  np.nan, info['eta0_avg'], info['eta0_max'], info['eta1_avg'], info['eta1_max'], m_d_error, m_error, tebd_error)
		print
	
	E_nrm+= H_nrm #How much norm/site changed
	T+= 3.5/2. #time accrued

	for j in range(N-1):
		PEPs, info = peps_ESWN_tebd(PEPs, [Uv, Uh, Uv, Uh], truncation_par, Os = [H[1], H[0], H[1], H[0]])
		m_error = info['moses_error']/V
		m_d_error = info['moses_d_error']/V
		tebd_error = info['tebd_error']/V
		H_exp = np.sum(info['expectation_O'])/V
		H_nrm = -np.log(info['nrm'])/V/dt

		if verbose:
			print ("{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}").format('Energy(nrm)', '<H>', '<eta0>', 'max(eta0)', '<eta1>', 'max(eta1)', 'moses_d_err', 'moses_err', 'tebd_err')
			print ("{:>12.7f}{:>12.7f}{:>12.2f}{:>12.2f}{:>12.2f}{:>12.2f}{:>12.2e}{:>12.2e}{:>12.2e}").format(H_nrm/2,  H_exp/2, info['eta0_avg'], info['eta0_max'], info['eta1_avg'], info['eta1_max'], m_d_error, m_error, tebd_error)
			print
		
		E_nrm+=H_nrm
		#E_exp+=H_exp
		T+=4/2.
	
	PEPs, info = peps_ESWN_tebd(PEPs, [Uv2, None, None, None], truncation_par, Os = [None, None, H[1], H[0]])
	m_error = info['moses_error']
	m_d_error = info['moses_d_error']
	tebd_error = info['tebd_error']
	E_exp =np.sum(info['expectation_O'][2:])
	H_nrm = -np.log(info['nrm'])/dt
	if verbose:
		print ("{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}").format('Energy(nrm)', '<H>', '<eta0>', 'max(eta0)', '<eta1>', 'max(eta1)', 'moses_d_err', 'moses_err', 'tebd_err')
		print ("{:>12.7f}{:>12.7f}{:>12.2f}{:>12.2f}{:>12.2f}{:>12.2f}{:>12.2e}{:>12.2e}{:>12.2e}").format(np.nan,  E_exp, info['eta0_avg'], info['eta0_max'], info['eta1_avg'], info['eta1_max'], m_d_error, m_error, tebd_error)
		print

	E_nrm+=-H_nrm
	T+=0.5/2.
	
	return PEPs, E_exp
