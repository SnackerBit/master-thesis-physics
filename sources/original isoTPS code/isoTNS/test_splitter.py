import numpy as np
from MZsplitter import split_psi, split_quad

np.random.seed(0)

def entanglement_entropy(psi):
	s = np.linalg.svd(psi,compute_uv=0)
	s=s[s>10**(-20)]**2        
	return -np.inner(np.log(s),s)	

def rand_u(D,u):
	X = np.eye(D) + u*(0.5-np.random.rand(D,D))
	Q,R = np.linalg.qr(X)
	return Q


#### Set dimensions 
chiL = 20
chiR = 20
dL = 4
dR = 4

#### Set "amount" of entangling
u_AI = 0.1
u_JB = 0.1
u_AB = 0.1
u_IJ = 1.

#### Created random product state in A,I,J,B
A = 0.5-np.random.rand(chiL)
I = 0.5-np.random.rand(dL)
J = 0.5-np.random.rand(dR)
B = 0.5-np.random.rand(chiR)
psi = A[:,None,None,None]*I[:,None,None]*J[:,None]*B
psi = psi / np.linalg.norm(psi)

#### Add entanglement
U_AI = rand_u(chiL*dL,u_AI).reshape([chiL,dL,chiL,dL])
U_JB = rand_u(dR*chiR,u_JB).reshape([dR,chiR,dR,chiR])
U_AB = rand_u(chiL*chiR,u_AB).reshape([chiL,chiR,chiL,chiR])
U_IJ = rand_u(dL*dR,u_IJ).reshape([dL,dR,dL,dR])

psi = np.tensordot(U_AB,psi,axes = ([2,3],[0,3])).transpose([0,2,3,1])
print "U_(AB): S_(AI)(JB) = ", entanglement_entropy(psi.reshape([chiL*dL,dR*chiR]))
psi = np.tensordot(U_IJ,psi,axes = ([2,3],[1,2])).transpose([2,0,1,3])
print "U_(IJ): S_(AI)(JB) = ", entanglement_entropy(psi.reshape([chiL*dL,dR*chiR]))
psi = np.tensordot(U_AI,psi,axes = ([2,3],[0,1]))
print "U_(AI): S_(AI)(JB) = ", entanglement_entropy(psi.reshape([chiL*dL,dR*chiR]))
psi = np.tensordot(psi,U_JB,axes = ([2,3],[2,3]))
print "U_(JB): S_(AI)(JB) = ", entanglement_entropy(psi.reshape([chiL*dL,dR*chiR]))

#### Apply splitter
a2, S2, B2, info = split_psi(psi.transpose(1,2,0,3).reshape(dL*dR,chiL,chiR), dL, dR)
a, S, B, info = split_quad(psi.transpose(1,2,0,3).reshape(dL*dR,chiL,1, chiR), dL, dR)

print "diff", np.linalg.norm(a - a2)

print 
s=info['s_Lambda'][info['s_Lambda']>10**(-20)]**2        
print  "After:  S = ",-np.inner(np.log(s),s)	
