""" Exact diagonalization code to find the ground state of 
a 1D quantum Ising model."""

import scipy.sparse as sparse 
import numpy as np 
import scipy.sparse.linalg.eigen.arpack as arp
import pylab as pl
import os.path

def gen_spin_operators(L): 
	"""" Returns the spin operators sigma_x and sigma_z for L sites """
	sx = sparse.csr_matrix(np.array([[0.,1.],[1.,0.]]))
	sz = sparse.csr_matrix(np.array([[1.,0.],[0.,-1.]]))
	    
	d = 2
	sx_list = []
	sz_list = []

	for i_site in range(L): 
		if i_site==0: 
			X=sx
			Z=sz 
		else: 
			X= sparse.csr_matrix(np.eye(d)) 
			Z= sparse.csr_matrix(np.eye(d))
		for j_site in range(1,L): 
			if j_site==i_site: 
				X=sparse.kron(X,sx, 'csr')
				Z=sparse.kron(Z,sz, 'csr') 
			else: 
				X=sparse.kron(X,np.eye(d),'csr') 
				Z=sparse.kron(Z,np.eye(d),'csr') 
		sx_list.append(X)
		sz_list.append(Z) 
		
	return sx_list,sz_list 

def gen_hamiltonian(sx_list,sz_list,Lx,Ly): 
	"""" Generates the Hamiltonian """    
	L = Lx*Ly
	H_zz = sparse.csr_matrix((2**L,2**L))
	H_x = sparse.csr_matrix((2**L,2**L))
	for i in range(Lx*Ly):
		H_x = H_x + sx_list[i]
		
	for x in range(Lx-1):
		for y in range(Ly-1):
			H_zz = H_zz + sz_list[x%Lx + (y%Ly)*Lx]*sz_list[(x+1)%Lx + (y%Ly)*Lx]
			H_zz = H_zz + sz_list[x%Lx + (y%Ly)*Lx]*sz_list[(x)%Lx + ((y+1)%Ly)*Lx]
			if x == Lx-2:
				H_zz = H_zz + sz_list[(x+1)%Lx + (y%Ly)*Lx]*sz_list[(x+1)%Lx + ((y+1)%Ly)*Lx]
			if y == Ly-2:
				H_zz = H_zz + sz_list[x%Lx + ((y+1)%Ly)*Lx]*sz_list[(x+1)%Lx + ((y+1)%Ly)*Lx]
				
	return H_zz, H_x 
	
def energy(J,g,L):
	# Set parameters here
	sx_list,sz_list  = gen_spin_operators(L*L)
	H_zz, H_x = gen_hamiltonian(sx_list,sz_list,L,L)
	fn = "ising_ed_J_%.2f"%J + "_g_%.2f"%g+ "_L_%.2f"%L
	
	if os.path.exists(fn):
		e = np.loadtxt(fn)
	else:
		e = arp.eigsh(J*H_zz + g*H_x,k=1,which='SA',return_eigenvectors=False)
		np.savetxt(fn,e)
		e = e.item()
	
	return e

print(energy(1,3.1,4))