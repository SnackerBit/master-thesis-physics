from mosesmove import *

#Test-data Weak inter-chain 2-leg TFI ladder in paramagnetic phase
#with open('test_data/2TFI_eps-0.5.mps', 'r') as f:
#with open('test_data/2HAF_h-0.05_J2-0.5.mps', 'r') as f:

#with open('test_data/i2TFI_eps-1.mps', 'r') as f:
#	Psi = cPickle.load(f)


"""This code takes a 2-leg TFI ladder, and decomposes it as

	Psi = A0 A1 A2 Lambda3
	
	weeeeeee
"""

np.set_printoptions(precision = 10, suppress = True, linewidth=120)

with open('test_data/3TFI_JH0.5.mps', 'r') as f:
#with open('test_data/2HAF_h-0.05_J2-0.1.mps', 'r') as f:
	Psi = cPickle.load(f)

print np.max( [b.shape[3] for b in Psi])
truncation_par = {'chi_max': {'eta0_max':16, 'eta1_max': 32, 'chiV_max':4, 'chiH_max':4}, 'p_trunc':1e-8 }
A0, Lambda1, info = sweeped_moses_move(Psi, truncation_par, save_info = './TFI/n2.0')
#A0, Lambda1, info = moses_move(Psi, truncation_par, save_info = './TFI/n2.0')
#print np.cumsum(info['errors'])
print [l.shape[3] for l in Lambda1], [l.shape[3] for l in A0], [l.shape[1] for l in A0]
print check_overlap(Psi, A0, Lambda1)

#Psi1 = peel(Lambda1, 2)
