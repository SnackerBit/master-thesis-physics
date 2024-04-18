from misc import *

def svd_theta(theta,eta):
	try:
		U, s, V = svd( theta, compute_uv=True, full_matrices=False)
	except np.linalg.linalg.LinAlgError:
		print "*dgesvd*"
		U, s, V = svd_dgesvd.svd_dgesvd(theta, full_matrices = 0, compute_uv = 1)

	nrm = np.linalg.norm(s)
	eta_new = np.min([ np.count_nonzero((1 - np.cumsum(s**2)/nrm**2) > 10e-16)+1, eta])
	nrm_t = np.linalg.norm(s[:eta_new])
	
	return U[:,:eta_new],s[:eta_new]/nrm_t,V[:eta_new,:], eta_new

def var_Lambda_1site(Psi,A,Lambda):
	L = len(Psi)
	Lp = np.zeros([1,1,1]);Lp[0,0,0] = 1.
	Lp_list = [Lp]
	
	for i in range(L):
		Lp = np.tensordot(Lp,Psi[i],axes = (0,2))
		Lp = np.tensordot(Lp,A[i], axes = ([1,2],[2,0]))
		Lp = np.tensordot(Lp,Lambda[i], axes = ([0,3,1],[2,0,1]))
		Lp = Lp.transpose([0,2,1])
		Lp_list.append(Lp)
	
	#### Left move		
	Rp = np.zeros([1,1,1]);Rp[0,0,0] = 1.
	Rp_list = [Rp]
	Lambdap = [ [] for i in range(L) ]

	for i in range(L-1,-1,-1):
		Rp = np.tensordot(Rp,Psi[i],axes=(0,3))
		Rp = np.tensordot(Rp,A[i],axes=([2,1],[0,3]))
		theta = np.tensordot(Rp,Lp_list[i],axes = ([2,4],[0,2]))
		theta = theta.transpose(2,1,3,0)
		
		d1,d2,chi1,chi2 = theta.shape
		Q,R = np.linalg.qr(theta.transpose(0,1,3,2).reshape(d1*d2*chi2,chi1), mode='raw')
		Lambdap[i] = Q.reshape(d1,d2,chi2,chi1).transpose(0,1,3,2)

		Rp = np.tensordot(Rp,Lambdap[i],axes=([0,1,3],[3,1,0]))
		Rp = Rp.transpose(0,2,1)
		Rp_list.append(Rp)
		
	#### Right move			
	Lp = np.zeros([1,1,1]);Lp[0,0,0] = 1.
	Lp_list = [Lp]
	Lambdap = [ [] for i in range(L) ]
	
	for i in range(L):
		Lp = np.tensordot(Lp,Psi[i],axes = (0,2))
		Lp = np.tensordot(Lp,A[i], axes = ([1,2],[2,0]))
		theta = np.tensordot(Lp,Rp_list[L-1-i], axes = ([2,4],[0,2]))
		theta = theta.transpose(2,1,0,3)
		
		d1,d2,chi1,chi2 = theta.shape
		Q,R = np.linalg.qr(theta.reshape(d1*d2*chi1,chi2))
		Lambdap[i] = Q.reshape(d1,d2,chi1,chi2)
		
		Lp = np.tensordot(Lp,Lambdap[i],axes=([0,1,3],[2,1,0]))
		Lp = Lp.transpose(0,2,1)
		Lp_list.append(Lp)
		
	return Lambdap
	
def var_Lambda_2site(Psi,A,Lambda,eta):
	L = len(Psi)
	Lp = np.zeros([1,1,1]);Lp[0,0,0] = 1.
	Lp_list = [Lp]
	
	for i in range(L):
		Lp = np.tensordot(Lp,Psi[i],axes = (0,2))
		Lp = np.tensordot(Lp,A[i], axes = ([1,2],[2,0]))
		Lp = np.tensordot(Lp,Lambda[i], axes = ([0,3,1],[2,0,1]))
		Lp = Lp.transpose([0,2,1])
		Lp_list.append(Lp)
	
	#### Left move		
	Rp = np.zeros([1,1,1]);Rp[0,0,0] = 1.
	Rp_list = [Rp]
	Lambdap = [ [] for i in range(L) ]

	for i in range(L-1,0,-1):
		Rp = np.tensordot(Rp,Psi[i],axes=(0,3))
		Rp = np.tensordot(Rp,A[i],axes=([2,1],[0,3]))
		
		theta = np.tensordot(Psi[i-1],Rp,axes=(3,2))
		theta = np.tensordot(theta,A[i-1],axes=([0,6],[0,3]))
		
		theta = np.tensordot(theta,Lp_list[i-1],axes = ([1,6],[0,2]))
		theta = theta.transpose(0,4,5,2,3,1)
		d1a,d2a,chia,d1b,d2b,chib = theta.shape
		theta = theta.reshape(d1a*d2a*chia,d1b*d2b*chib)
		
		U,s,V,eta_new = svd_theta(theta,eta)
		
		M  = V.reshape(eta_new,d1b,d2b,chib).transpose(2,1,0,3)		
		Rp = np.tensordot(Rp,M,axes=([0,1,3],[3,1,0]))
		Rp = Rp.transpose(0,2,1)
		Rp_list.append(Rp)
		
	#### Right move
	Lp = np.zeros([1,1,1]);Lp[0,0,0] = 1.
	Lp_list = [Lp]
	Lambdap = [ [] for i in range(L) ]

	for i in range(L-1):
		Lp = np.tensordot(Lp,Psi[i],axes = (0,2))
		Lp = np.tensordot(Lp,A[i], axes = ([1,2],[2,0]))

		theta = np.tensordot(Lp,Psi[i+1],axes=(2,2))
		theta = np.tensordot(theta,A[i+1],axes=([3,4],[2,0]))
		theta = np.tensordot(theta,Rp_list[L-2-i], axes = ([4,6],[0,2]))
		theta = theta.transpose(1,2,0,3,4,5)

		d1a,d2a,chia,d1b,d2b,chib = theta.shape
		theta = theta.reshape(d1a*d2a*chia,d1b*d2b*chib)
		
		U,s,V,eta_new = svd_theta(theta,eta)

		M  = U.reshape(d1a,d2a,chia,eta_new).transpose(1,0,2,3)
		Lambdap[i] = M

		Lp = np.tensordot(Lp,M,axes=([0,1,3],[2,1,0]))
		Lp = Lp.transpose(0,2,1)
		Lp_list.append(Lp)
		
	M = np.dot(np.diag(s),V).reshape(eta_new,d1b,d2b,chib).transpose(2,1,0,3)
	Lambdap[L-1] = M
		
	return Lambdap
	
def var_A(Psi,A,Lambda):
	L = len(Psi)
	Lp = np.zeros([1,1,1]);Lp[0,0,0] = 1.
	Lp_list = [Lp]
	
	for i in range(L):
		Lp = np.tensordot(Lp,Psi[i],axes = (0,2))
		Lp = np.tensordot(Lp,A[i], axes = ([1,2],[2,0]))
		Lp = np.tensordot(Lp,Lambda[i], axes = ([0,3,1],[2,0,1]))
		Lp = Lp.transpose([0,2,1])
		Lp_list.append(Lp)
	
	#### Left move		
	Rp = np.zeros([1,1,1]);Rp[0,0,0] = 1.
	Rp_list = [Rp]
	Ap = [ [] for i in range(L) ]

	for i in range(L-1,-1,-1):
		Rp = np.tensordot(Rp,Psi[i],axes=(0,3))
		Rp = np.tensordot(Rp,Lambda[i],axes=([0,3],[3,1]))
		theta = np.tensordot(Rp,Lp_list[i],axes = ([2,4],[0,1]))
		
		theta = theta.transpose(1,3,2,0)
		
		chiL,chiD,chiR,chiU = theta.shape 
		X,s,Y = np.linalg.svd(theta.reshape(chiL*chiD,chiR*chiU),full_matrices=False)
		Ap[i] = np.dot(X,Y).reshape(chiL,chiD,chiR,chiU).transpose(0,2,1,3)
	
		Rp = np.tensordot(Rp,Ap[i],axes=([0,1,3],[3,0,1]))
		Rp_list.append(Rp)
	
	return Ap
	
def var_moses(Psi,A,Lambda,eta,N = 10):
	for i in range(N):
		A = var_A(Psi,A,Lambda)
		Lambda = var_Lambda_2site(Psi,A,Lambda,eta)
	return A, Lambda
