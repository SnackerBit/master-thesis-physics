import cPickle
from misc import *
from mosesmove import moses_move
from var_splitter import var_moses
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
									

def SvN(s):
	"von Neumann entropy"
	p = s**2
	p[p<1e-14] = 1e-14
	
	return -np.vdot(p, np.log(p))
	

if __name__ == "__main__": 
	
	chi_h = 4
	chi_v = 4
	eta_list = 	[3,4,5,6,7,8,9,10]
	fn = './test_data/2TFI_eps-0.5.mps'
	
	with open(fn, 'r') as f:
		Psi = cPickle.load(f)
			
	dPsi_MM = []
	dPsi_V = []
	for eta in eta_list:
		A, Lambda, info = sweeped_moses_move(Psi,truncation_par={'chi_max': {'eta1_max': eta, 'eta0_max':eta,'etaV_max':eta,'etaH_max':eta, 'chiV_max':chi_v, 'chiH_max':chi_h}, 'p_trunc':1e-10 })
		dPsi_MM.append(1-overlap(Psi,A,Lambda))
		print "moses      ", dPsi_MM[-1],
		
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
