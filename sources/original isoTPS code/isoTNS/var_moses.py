import cPickle
from misc import *
from mosesmove import moses_move
from mosesmove import sweeped_moses_move
import pylab as pl
np.set_printoptions(linewidth=2000, precision=1,threshold=4000)
np.random.seed(0)

def overlap(Psi,A,Lambda):
	L = len(Psi)
	O = np.ones([1,1])
	N = np.ones([1,1])
	for i in np.arange(L):
		d1,d2,chi1,chi2 = Psi[i].shape
		B = np.reshape(Psi[i],(d1*d2,chi1,chi2))
		C = np.tensordot(A[i],Lambda[i],axes=[1,0])
		C = np.transpose(C,(0,3,1,4,2,5))
		d,d,chiA_1,chiL_1,chiA_2,chiL_2 = C.shape
		C = np.reshape(C,(d1*d2,chiA_1*chiL_1,chiA_2*chiL_2))
		
		O = np.tensordot(O,np.conj(B), axes=(1,1))
		O = np.tensordot(O,C, axes=([0,1],[1,0]))
		O = np.transpose(O,(1,0))
		
		N = np.tensordot(N,np.conj(C), axes=(1,1))
		N = np.tensordot(N,C, axes=([0,1],[1,0]))
		N = np.transpose(N,(1,0))

	O = np.trace(O)
	N = np.trace(N)	
	return O/N
	
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
	
def mpo_to_full(A):
	D = A[0].shape[2]
	vL = np.zeros(D)
	vL[0] = 1.
	vR = np.zeros(D)
	vR[D-1] = 1.
	
	L = len(A)
	
	d1 =  A[0].shape[0]
	d2 =  A[0].shape[1]

	A_full = np.tensordot(vL,A[0].transpose(2,3,0,1),axes=(0,0))
	for i in range(0,L-1):
		A_full = np.tensordot(A_full,A[i+1].transpose(2,3,0,1),axes=(2*i,0))
		d1 =  d1*A[i+1].shape[0]
		d2 =  d2*A[i+1].shape[1]
	
	print d1,d2
	A_full = np.tensordot(A_full,vR,axes=(L*2-2,0))
	A_full=np.transpose(A_full,np.hstack([np.arange(0,L*2,2),np.arange(0,L*2,2)+1]))
	A_full=np.reshape(A_full,[d1,d2])
	
	return A_full
									
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
	
def SvN(s):
	"von Neumann entropy"
	p = s**2
	p[p<1e-14] = 1e-14
	
	return -np.vdot(p, np.log(p))
	
def random_mps(d1,d2,chi,L, form = 'B'):
	Psi = []
	for i in range(L):
		chi1 = np.min([(d1*d2)**np.min([i,L-i]),chi])
		chi2 = np.min([(d1*d2)**np.min([i+1,L-i-1]),chi])
		Psi.append(0.5 - np.random.rand(d1,d2,chi1,chi2))
	
	if form == 'B':
		for i in range(L-1,-1,-1):
			d1,d2,chi1,chi2 = Psi[i].shape
			Q,R = np.linalg.qr(Psi[i].transpose(0,1,3,2).reshape(d1*d2*chi2,chi1))
			Psi[i] = Q.reshape(d1,d2,chi2,chi1).transpose(0,1,3,2)
			if i>0:
				Psi[i-1]=np.tensordot(Psi[i-1],R,axes=(3,1))
	return Psi

def var_moses(Psi,A,Lambda,eta,N = 10):
	for i in range(N):
		A = var_A(Psi,A,Lambda)
		Lambda = var_Lambda_2site(Psi,A,Lambda,eta)
	return A, Lambda

if __name__ == "__main__": 
	
	chi_h = 2
	chi_v = 2
	eta_list = 	[6]
	fn = './test_data/2HAF_h-0.05_J2-0.9.mps'
	fn = './test_data/2TFI_eps-0.5.mps'
	
	with open(fn, 'r') as f:
		Psi = cPickle.load(f) 
			
	dPsi_MM = []
	dPsi_V = []
	for eta in eta_list:
		A, Lambda, info = sweeped_moses_move(Psi,truncation_par={'chi_max': {'eta1_max': eta, 'eta0_max':eta, 'chiV_max':chi_v, 'chiH_max':chi_h}, 'p_trunc':1e-10 })
		dPsi_MM.append(1-overlap(Psi,A,Lambda))
		print "moses      ", dPsi_MM[-1],
		
		for a in A:
			print a.shape
		exit()
		A, Lambda = var_moses(Psi,A,Lambda,eta, N = 200*eta)

		dPsi_V.append(1-overlap(Psi,A,Lambda))		
		print " --> ", dPsi_V[-1]
	
	pl.semilogy(eta_list,dPsi_MM,'-o')
	pl.semilogy(eta_list,dPsi_V,'-o')
	pl.ylabel('$1-\\langle\\psi|A,\\Lambda\\rangle$')
	pl.xlabel('$\\eta$')
	pl.title(fn)
	pl.legend(['Moses','Variational'])
	pl.show()
