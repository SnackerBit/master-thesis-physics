from tebd import *

def run_tfi(J,g,L,chi,f,e_ed):
	eta = f*chi
	Lx = L
	Ly = L
	
	print "J =",J,"g =",g,"Lx =",Lx,"Ly =",Ly, "E_ed = ",e_ed
	print "chi =",chi,"eta=",eta
	
	truncation_par = {'chi_max':{'eta0_max':eta, 'chiV_max':chi, 'chiH_max':chi, 'eta1_max':eta},'p_trunc':1e-8}

	t = (0.5-np.random.rand(2, 1, 1, 1, 1))/100.
	t = t/np.linalg.norm(t)

	Hh = H_TFI(L, g/2, J)
	Hv = H_TFI(L, g/2, J)
	
	PEPs = [ [t.copy() for y in range(L)] for x in range(L)]
	H = (Hh, Hv)
	
	dts = 1.5*np.exp(-0.5*np.arange(1, 12))
	
	Tstep = 1.5
	es = []
	for dt in dts:
		PEPs, e = p2_tebd(PEPs, H, dt, int(Tstep/dt), truncation_par)
		PEPs, e = p2_tebd(PEPs, H, dt, 1, truncation_par)
		print "%03.5f"%dt, "%.8f"%e,(e-e_ed)/np.abs(e_ed)
		es.append(e)
		np.savetxt("ising_moses_J_%.2f"%J + "_g_%.2f"%g+ "_L_%.0f"%L + "_eta_%.0f"%eta + "_chi_%.0f"%chi,np.vstack([dts[:len(es)],es]))
		
	print "E(dt)", es

if __name__ == "__main__":

	e_ed = {4:-51.701896,5:-81.038243, 6:-116.9438125677, 8:-208.4631854858, 10:-326.260419}
	J = 1
	g = 3.5
	chi = int(sys.argv[1])
	f = int(sys.argv[2])
	L = int(sys.argv[3])
        run_tfi(J,g,L,chi,f,e_ed[L])
