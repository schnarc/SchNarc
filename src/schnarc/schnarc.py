import numpy as np


def get_schnarc(prediction,properties):
    n_states = prediction['energy'][0].shape[0]
    QMout={}
    for i,prop in enumerate(properties):
        if prop == "energy":
            hamiltonian = np.zeros((n_states,n_states),dtype=complex)
            np.fill_diagonal(hamiltonian,prediction['energy'][0])
            hamiltonian_list = [ [hamiltonian[i][j] for i in range(n_states) ] for j in range(n_states) ]
            QMout['h'] = hamiltonian_list
            if QMout['hessian']:
              n_atoms = prediction['hessian'][0].shape[1]
              nonadiabatic_couplings = np.zeros((n_states,n_states,n_atoms,3))
              iterator = -1
              for istate in range(n_states):
                  for jstate in range(istate+1,n_states):
                      iterator+=1
                      nonadiabatic_couplings[istate][jstate] = prediction['nacs'][0][iterator]
                      nonadiabatic_couplings[jstate][istate] = -prediction['nacs'][0][iterator]
              QMout['nacdr'] = nonadiabatic_couplings
 
        elif prop == "force":
            QMout['grad'] = prediction['force'][0]

        elif prop == "socs":
            soc_matrix = np.zeros((n_states,n_states),dtype=complex)

            for istate in range(n_states):
                for jstate in range(n_states):
                    soc_matrix[istate][jstate] = prediction['socs'][istate][jstate]

        elif prop == "dipoles":
            dipole_matrix = np.zeros((3, n_states, n_states), dtype=complex)
            for xyz in range(3):
                iterator=-1
                for istate in range(n_states):
                    for jstate in range(istate,n_states):
                        iterator+=1
                        dipole_matrix[xyz][istate][jstate] = prediction['dipoles'][0][iterator][xyz]
                        dipole_matrix[xyz][jstate][istate] = prediction['dipoles'][0][iterator][xyz]
            QMout['dm'] = dipole_matrix

        elif prop == "nacs":
            n_atoms = prediction['nacs'][0].shape[1]
            nonadiabatic_couplings = np.zeros((n_states,n_states,n_atoms,3))
            iterator = -1
            for istate in range(n_states):
                for jstate in range(istate+1,n_states):
                    iterator+=1
                    nonadiabatic_couplings[istate][jstate] = prediction['nacs'][0][iterator]
                    nonadiabatic_couplings[jstate][istate] = -prediction['nacs'][0][iterator]
            QMout['nacdr'] = nonadiabatic_couplings
    return QMout

def get_socs(socs,n_states,energy):
    n_singlets=n_states['n_singlets']
    n_triplets=n_states['n_triplets']
    n_states = n_states['n_states']
    h_string = "! 1 Hamiltonian Matrix (%dx%d), complex \n %d %d \n"      % ( n_states,n_states,n_states,n_states)
    hamiltonian = np.zeros((n_states,n_states),dtype=complex)
    if n_triplets==int(0):
        index=np.argsort(energy)
    else:
        index=np.argsort(energy[0:n_singlets]) #ange(n_singlets+3*n_triplets)
        indext=np.argsort(energy[n_singlets:int(n_singlets+n_triplets)]) #ange(n_singlets+3*n_triplets)
    iterator=0
    for i in range(n_states):
      for j in range(i+1,n_states):
        hamiltonian[i][j] = (socs[2*iterator]+socs[2*iterator+1]*1j)
        iterator+=1
    hamiltonian = hamiltonian+hamiltonian.conj().T
    print(n_states)
    for i in range(n_singlets):
        hamiltonian[i][i] = (energy[index[i]])
    for i in range(n_singlets,n_singlets+n_triplets):
        hamiltonian[i][i] = energy[indext[i-n_singlets]+n_singlets]
        hamiltonian[i+n_triplets][i+n_triplets]=energy[indext[i-n_singlets]+n_singlets]
        hamiltonian[i+2*n_triplets][i+2*n_triplets]=energy[indext[i-n_singlets]+n_singlets]

    for istate in hamiltonian:
        for jstate in istate:
            h_string += "%20.12E %20.12E "%(np.real(jstate), np.imag(jstate))
        h_string+="\n"
    h_string += "\n"
    return h_string

def get_diab(coupling,energy,n_states):
    n_singlets=n_states['n_singlets']
    n_triplets=n_states['n_triplets']
    n_states = n_states['n_states']
    h_string = "! 1 Hamiltonian Matrix (%dx%d), complex \n %d %d \n"      % (n_states,n_states,n_states,n_states)
    hamiltonian = np.zeros((n_states,n_states),dtype=complex)
    np.fill_diagonal(hamiltonian,energy)
    iterator=0
    for icoupling in range(energy.shape[0]):
        for jcoupling in range(icoupling+1,energy.shape[0]):
            hamiltonian[icoupling][jcoupling] = coupling[iterator]#*(energy[icoupling]-energy[jcoupling])
            hamiltonian[jcoupling][icoupling] = coupling[iterator]#*(energy[icoupling]-energy[jcoupling])
            iterator+=1
    for istate in hamiltonian:
        for jstate in istate:
            h_string += "%20.12E %20.12E "%(np.real(jstate), np.imag(jstate))
        h_string += "\n"
    h_string += "\n"
    return h_string

def get_energy(energy,n_states):
    n_singlets= n_states['n_singlets']
    n_triplets= n_states['n_triplets']
    n_states =  n_states['n_states']
    h_string = "! 1 Hamiltonian Matrix (%dx%d), complex \n %d %d \n"      % ( n_states,n_states,n_states,n_states)
    hamiltonian = np.zeros((n_states,n_states),dtype=complex)
    if n_triplets==int(0):
        index=np.argsort(energy)
    else:
        index=np.argsort(energy[0:n_singlets]) #ange(n_singlets+3*n_triplets)
        indext=np.argsort(energy[n_singlets:int(n_singlets+n_triplets)]) #ange(n_singlets+3*n_triplets)
    for i in range(n_singlets):
        hamiltonian[i][i] = (energy[index[i]])
    for i in range(n_triplets):
        hamiltonian[i+n_singlets][n_singlets+i] = energy[indext[i]+n_singlets]
        hamiltonian[i+n_singlets+n_triplets][i+n_singlets+n_triplets]=energy[indext[i]+n_singlets]
        hamiltonian[i+n_singlets+2*n_triplets][i+n_singlets+2*n_triplets]=energy[indext[i]+n_singlets]
    for istate in hamiltonian:
        for jstate in istate:
            h_string += "%20.12E %20.12E "%(np.real(jstate), np.imag(jstate))
        h_string+="\n"
    h_string += "\n"
    return h_string

def get_force(force,n_states,energy):
    n_singlets= n_states['n_singlets']
    n_triplets=n_states['n_triplets']
    n_states = n_states['n_states']
    n_atoms = force.shape[1]
    if n_triplets==int(0):
        index=np.argsort(energy)
    else:
        index=np.argsort(energy[0:n_singlets])
        indext=np.argsort(energy[n_singlets:(n_singlets+n_triplets)])
    g_string= "! 3 Gradient Vectors (%dx%dx3, real) \n" % (n_states, n_atoms)
    for istate in range(n_singlets):
        g_string += "%d %d ! state %d \n" %(n_atoms, 3, istate+1)
        for iatom in range(n_atoms):
            for xyz in range(3):
                g_string += "%20.12E " %-force[index[istate]][iatom][xyz]
            g_string += "\n"
    for itriplet in range(3):
        for istate in range(n_triplets):
            g_string += "%d %d ! state %d \n" %(n_atoms, 3, (istate+n_triplets*itriplet)+1)
            for iatom in range(n_atoms):
                for xyz in range(3):
                    g_string += "%20.12E " %-force[indext[istate]+n_singlets][iatom][xyz]
                g_string += "\n"
    return g_string

def get_dipoles(dipoles,n_states,energy):
    n_singlets=n_states['n_singlets']
    n_triplets=n_states['n_triplets']
    n_states = n_states['n_states']
    dipole_matrix = np.zeros((3, n_states, n_states), dtype=complex)
    d_string="! 2 Dipole Moment Matrices (3x%sx%s, complex)\n" %(n_states,n_states)
    if n_triplets==int(0):
        index=np.argsort(energy)
    else:
        index= np.argsort(energy[0:n_singlets])
        indext=np.argsort(energy[n_singlets:(n_singlets+n_triplets)])
    for xyz in range(3):
        iterator=-1
        for istate in range(n_singlets):
            for jstate in range(istate,n_singlets):
                iterator+=1
                dipole_matrix[xyz][index[istate]][index[jstate]] = dipoles[iterator][xyz]
                dipole_matrix[xyz][index[jstate]][index[istate]] = dipoles[iterator][xyz]
        for istate in range(n_triplets):
            for jstate in range(istate,n_triplets):
                iterator+=1
                for itriplet in range(3):
                   dipole_matrix[xyz][indext[istate]+n_singlets+n_triplets*itriplet][indext[jstate]+n_singlets+n_triplets*itriplet] = dipoles[iterator][xyz]
                   dipole_matrix[xyz][indext[jstate]+n_singlets+n_triplets*itriplet][indext[istate]+n_singlets+n_triplets*itriplet] = dipoles[iterator][xyz]
        d_string+="%d %d \n" %(n_states,n_states)
        for istate in range(n_states):
            for jstate in range(n_states):
                d_string+="%20.12E %20.12E " %(np.real(dipole_matrix[xyz][istate][jstate]), np.imag(dipole_matrix[xyz][istate][jstate]))
            d_string+="\n"
    return d_string

def get_nacs_deltaH2(hessian,energy,forces,n_states):
    n_singlets=n_states['n_singlets']
    n_triplets=n_states['n_triplets']
    n_states = n_states['n_states']
    n_atoms = forces.shape[1]
    g_string="! 5 Non-adiabatic Couplings (ddr) (%dx%dx%dx3, real) \n" % (n_states, n_states, n_atoms)
    nonadiabatic_couplings = np.zeros((n_states,n_states,n_atoms,3))
    iterator = -1
    hopping_direction=np.zeros( (n_singlets + n_triplets,n_singlets + n_triplets, n_atoms, 3) )
    hopping_magnitude= np.zeros( (n_singlets + n_triplets, n_singlets + n_triplets,1) )
    for istate in range(n_singlets):
        for jstate in range(istate+1,n_singlets):
            Hi=hessian[istate]
            dE=(energy[istate]-energy[jstate])
            Hj=hessian[jstate]
            dH_2_ij = (Hi-Hj)
            magnitude = dH_2_ij

            #SVD
            u,s,vh=np.linalg.svd(magnitude)
            ev=vh[0]
            ew=s[0]
            iterator=-1

            for iatom in range(n_atoms):
                for xyz in range(3):
                    iterator+=1
                    hopping_direction[istate][jstate][iatom][xyz] = ev[iterator]
                    hopping_direction[jstate][istate][iatom][xyz] = -ev[iterator]
            hopping_magnitude[istate][jstate]=ew
            hopping_magnitude[jstate][istate]=ew

    for istate in range(n_singlets,n_singlets+n_triplets):
        for jstate in range(istate+1,n_singlets+n_triplets):
            Hi=hessian[istate]
            dE=np.abs(energy[istate]-energy[jstate])
            Hj=hessian[jstate]
            dH_2_ij = (Hi-Hj)
            magnitude = dH_2_ij
            iterator=-1
            u,s,vh = np.linalg.svd(magnitude)
            ev=vh[0]
            ew=s[0]
            iterator=-1
            for iatom in range(n_atoms):
                for xyz in range(3):
                    iterator+=1
                    hopping_direction[istate][jstate][iatom][xyz] = ev[iterator]
                    hopping_direction[jstate][istate][iatom][xyz] = -ev[iterator]
            hopping_magnitude[istate][jstate]=ew
            hopping_magnitude[jstate][istate]=ew
    threshold_dE = 100

    compute_nac=False
    for istate in range(n_singlets):
        for jstate in range(istate+1, n_singlets):
            if np.abs(energy[istate]-energy[jstate]) <= threshold_dE:
                #magnitude =np.sqrt(hopping_magnitude[istate][jstate])/np.sqrt(np.abs(energy[istate]-energy[jstate]))/2
                magnitude =(hopping_magnitude[istate][jstate])/(np.abs(energy[istate]-energy[jstate]))**2/2
                
                nonadiabatic_couplings[istate][jstate][:][:] = hopping_direction[istate][jstate] * magnitude
                nonadiabatic_couplings[jstate][istate][:][:] = -nonadiabatic_couplings[istate][jstate]
    for istate in range(n_singlets,n_singlets+n_triplets):
        for jstate in range(istate+1,n_singlets+n_triplets):
            if np.abs(energy[istate]-energy[jstate]) <= threshold_dE:
                for itriplet in range(3):
                    #magnitude = np.sqrt(hopping_magnitude[istate][jstate])/(np.abs(energy[istate]-energy[jstate]))/2
                    magnitude =(hopping_magnitude[istate][jstate])/(np.abs(energy[istate]-energy[jstate]))**2/2
                    nonadiabatic_couplings[istate+n_triplets*itriplet][jstate+n_triplets*itriplet]=hopping_direction[istate][jstate] * magnitude
                    nonadiabatic_couplings[istate+n_triplets*itriplet][jstate+n_triplets*itriplet] = -nonadiabatic_couplings[jstate+n_triplets*itriplet][istate+n_triplets*itriplet]
    for istate in range(n_states):
        for jstate in range(n_states):
            g_string+="State %d %d\n" %(istate+1,jstate+1)
            for iatom in range(n_atoms):
                for xyz in range(3):
                    g_string += "%20.12E " %(nonadiabatic_couplings[istate][jstate][iatom][xyz])
                g_string += "\n"
    return g_string
    
def get_nacs_deltaH3(hessian,energy,forces,n_states):
    n_singlets=n_states['n_singlets']
    n_triplets=n_states['n_triplets']
    n_states = n_states['n_states']
    n_atoms = forces.shape[1]
    g_string="! 5 Non-adiabatic Couplings (ddr) (%dx%dx%dx3, real) \n" % (n_states, n_states, n_atoms)
    nonadiabatic_couplings = np.zeros((n_states,n_states,n_atoms,3))
    iterator = -1
    forces = -forces
    hopping_direction=np.zeros( (n_singlets + n_triplets,n_singlets + n_triplets, n_atoms, 3) )
    hopping_magnitude= np.zeros( (n_singlets + n_triplets, n_singlets + n_triplets,1) )
    for istate in range(n_singlets):
        for jstate in range(istate+1,n_singlets):
            Hi=hessian[istate]
            dE=(energy[istate]-energy[jstate])
            Hj=hessian[jstate]
            GiGi=np.dot(-forces[istate].reshape(-1,1),-forces[istate].reshape(-1,1).T)
            GjGj=np.dot(-forces[jstate].reshape(-1,1),-forces[jstate].reshape(-1,1).T)
            GiGj=np.dot(-forces[istate].reshape(-1,1),-forces[jstate].reshape(-1,1).T)
            GjGi=np.dot(-forces[jstate].reshape(-1,1),-forces[istate].reshape(-1,1).T)
            G_diff = 0.5*(-forces[istate]+forces[jstate])
            G_diff2= np.dot(G_diff.reshape(-1,1),G_diff.reshape(-1,1).T)
            dH_2_ij = 0.5*(dE*(Hi-Hj) + GiGi + GjGj - GiGj-GjGi)
            magnitude = dH_2_ij/2-G_diff2

            #SVD
            u,s,vh=np.linalg.svd(magnitude)
            ev=vh[0]
            #get one phase
            e=max(ev[0:2].min(),ev[0:2].max(),key=abs)
            if e>=0.0:
                pass
            else:
                ev=-ev
            ew=s[0]
            iterator=-1
            for iatom in range(n_atoms):
                for xyz in range(3):
                    iterator+=1
                    hopping_direction[istate][jstate][iatom][xyz] = ev[iterator]
                    hopping_direction[jstate][istate][iatom][xyz] = -ev[iterator]
            hopping_magnitude[istate][jstate]=np.sqrt(ew)
            hopping_magnitude[jstate][istate]=np.sqrt(ew)

    for istate in range(n_singlets,n_singlets+n_triplets):
        for jstate in range(istate+1,n_singlets+n_triplets):
            Hi=hessian[istate]
            dE=np.abs(energy[istate]-energy[jstate])
            Hj=hessian[jstate]
            GiGi=np.dot(-forces[istate].reshape(-1,1),-forces[istate].reshape(-1,1).T)
            GjGj=np.dot(-forces[jstate].reshape(-1,1),-forces[jstate].reshape(-1,1).T)
            GiGj=np.dot(-forces[istate].reshape(-1,1),-forces[jstate].reshape(-1,1).T)
            GjGi=np.dot(-forces[jstate].reshape(-1,1),-forces[istate].reshape(-1,1).T)
            G_diff = 0.5*(-forces[istate]+forces[jstate])
            G_diff2= np.dot(G_diff.reshape(-1,1),G_diff.reshape(-1,1).T)
            dH_2_ij = 0.5*(dE*(Hi-Hj) + GiGi + GjGj - GiGj-GjGi)
            magnitude = dH_2_ij/2-G_diff2
            iterator=-1
            u,s,vh = np.linalg.svd(magnitude)
            ev=vh[0]
            #get one phase
            e=max(ev[0:2].min(),ev[0:2].max(),key=abs)
            if e>=0.0:
                pass
            else:
                ev=-ev
            ew=s[0]
            iterator=-1
            for iatom in range(n_atoms):
                for xyz in range(3):
                    iterator+=1
                    hopping_direction[istate][jstate][iatom][xyz] = ev[iterator]
                    hopping_direction[jstate][istate][iatom][xyz] = -ev[iterator]
            hopping_magnitude[istate][jstate]=np.sqrt(ew)
            hopping_magnitude[jstate][istate]=np.sqrt(ew)
    threshold_dE_S = 0.018
    threshold_dE_T = 0.036
    for istate in range(n_singlets):
        for jstate in range(istate+1, n_singlets):
            if np.abs(energy[istate]-energy[jstate]) <= threshold_dE_S:
                dE = np.abs(energy[istate]-energy[jstate])
                if dE == int(0):
                    dE = 0.00000001
                magnitude =((hopping_magnitude[istate][jstate]))/dE
                nonadiabatic_couplings[istate][jstate][:][:] = hopping_direction[istate][jstate] * magnitude
                nonadiabatic_couplings[jstate][istate][:][:] = -nonadiabatic_couplings[istate][jstate]
    for istate in range(n_singlets,n_singlets+n_triplets):
        for jstate in range(istate+1,n_singlets+n_triplets):
            if np.abs(energy[istate]-energy[jstate]) <= threshold_dE_T:
                dE = np.abs(energy[istate]-energy[jstate])
                if dE == int(0):
                    dE = 0.00000001
                for itriplet in range(3):
                    magnitude = (hopping_magnitude[istate][jstate])/dE
                    nonadiabatic_couplings[istate+n_triplets*itriplet][jstate+n_triplets*itriplet]=hopping_direction[istate][jstate] * magnitude
                    nonadiabatic_couplings[jstate+n_triplets*itriplet][istate+n_triplets*itriplet] = -nonadiabatic_couplings[istate+n_triplets*itriplet][jstate+n_triplets*itriplet]
    for istate in range(n_singlets+3*n_triplets):
        for jstate in range(n_singlets+3*n_triplets):
            g_string+="State %d %d\n" %(istate+1,jstate+1)
            for iatom in range(n_atoms):
                for xyz in range(3):
                    g_string += "%20.12E " %(nonadiabatic_couplings[istate][jstate][iatom][xyz])
                g_string += "\n"
    return g_string
def get_nacs_deltaH4(hessian,energy,forces,n_states):
    n_singlets=n_states['n_singlets']
    n_triplets=n_states['n_triplets']
    n_states = n_states['n_states']
    n_atoms = forces.shape[1]
    g_string="! 5 Non-adiabatic Couplings (ddr) (%dx%dx%dx3, real) \n" % (n_states, n_states, n_atoms)
    nonadiabatic_couplings = np.zeros((n_states,n_states,n_atoms,3))
    iterator = -1
    forces = -forces
    hopping_direction=np.zeros( (n_singlets + n_triplets,n_singlets + n_triplets, n_atoms, 3) )
    hopping_magnitude= np.zeros( (n_singlets + n_triplets, n_singlets + n_triplets,1) )
    hopping_magnitude2= np.zeros( (n_singlets + n_triplets, n_singlets + n_triplets,1) )
    for istate in range(n_singlets):
        for jstate in range(istate+1,n_singlets):
            Hi=hessian[istate]
            dE=(energy[istate]-energy[jstate])
            Hj=hessian[jstate]
            GiGi=np.dot(forces[istate].reshape(-1,1),forces[istate].reshape(-1,1).T)
            GjGj=np.dot(forces[jstate].reshape(-1,1),forces[jstate].reshape(-1,1).T)
            GiGj=np.dot(forces[istate].reshape(-1,1),forces[jstate].reshape(-1,1).T)
            GjGi=np.dot(forces[jstate].reshape(-1,1),forces[istate].reshape(-1,1).T)
            G_diff = 0.5*(forces[istate]-forces[jstate])
            G_diff2= np.dot(G_diff.reshape(-1,1),G_diff.reshape(-1,1).T)
            dH_2_ij = 0.5*(dE*(Hi-Hj) + GiGi + GjGj - 2*GiGj)
            magnitude = dH_2_ij-G_diff2
            magnitude_force = np.dot(G_diff.reshape(-1),G_diff.reshape(-1))
            #SVD
            u,s,vh=np.linalg.svd(magnitude)
            ev=vh[0]
            ew=s[0]
            iterator=-1

            for iatom in range(n_atoms):
                for xyz in range(3):
                    iterator+=1
                    hopping_direction[istate][jstate][iatom][xyz] = ev[iterator]
                    hopping_direction[jstate][istate][iatom][xyz] = -ev[iterator]
            hopping_magnitude2[istate][jstate]=magnitude_force
            hopping_magnitude2[jstate][istate]=magnitude_force
            hopping_magnitude[istate][jstate]=ew
            hopping_magnitude[jstate][istate]=ew

    for istate in range(n_singlets,n_singlets+n_triplets):
        for jstate in range(istate+1,n_singlets+n_triplets):
            Hi=hessian[istate]
            dE=np.abs(energy[istate]-energy[jstate])
            Hj=hessian[jstate]
            GiGi=np.dot(forces[istate].reshape(-1,1),forces[istate].reshape(-1,1).T)
            GjGj=np.dot(forces[jstate].reshape(-1,1),forces[jstate].reshape(-1,1).T)
            GiGj=np.dot(forces[istate].reshape(-1,1),forces[jstate].reshape(-1,1).T)
            GjGi=np.dot(forces[jstate].reshape(-1,1),forces[istate].reshape(-1,1).T)
            G_diff = 0.5*(forces[istate]-forces[jstate])
            G_diff2= np.dot(G_diff.reshape(-1,1),G_diff.reshape(-1,1).T)
            dH_2_ij = 0.5*(dE*(Hi-Hj) + GiGi + GjGj - 2*GiGj)
            magnitude = dH_2_ij/2-G_diff2
            magnitude_force = np.dot(G_diff.reshape(-1),G_diff.reshape(-1))
            iterator=-1
            u,s,vh = np.linalg.svd(magnitude)
            ev=vh[0]
            ew=s[0]
            iterator=-1
            for iatom in range(n_atoms):
                for xyz in range(3):
                    iterator+=1
                    hopping_direction[istate][jstate][iatom][xyz] = ev[iterator]
                    hopping_direction[jstate][istate][iatom][xyz] = -ev[iterator]
            hopping_magnitude[istate][jstate]=ew
            hopping_magnitude[jstate][istate]=ew
            hopping_magnitude2[istate][jstate]=magnitude_force
            hopping_magnitude2[jstate][istate]=magnitude_force
    threshold_dE = 0.04

    compute_nac=False
    for istate in range(n_singlets):
        for jstate in range(istate+1, n_singlets):
            if np.abs(energy[istate]-energy[jstate]) <= threshold_dE:
                magnitude =np.sqrt(hopping_magnitude[istate][jstate])/(np.sqrt(np.abs(energy[istate]-energy[jstate])))/2
                magnitude2= np.sqrt(hopping_magnitude2[istate][jstate]**2/(np.abs(energy[istate]-energy[jstate])**2))/2
                nonadiabatic_couplings[istate][jstate][:][:] = hopping_direction[istate][jstate] * (magnitude+magnitude2)
                nonadiabatic_couplings[jstate][istate][:][:] = -nonadiabatic_couplings[istate][jstate]
    for istate in range(n_singlets,n_singlets+n_triplets):
        for jstate in range(istate+1,n_singlets+n_triplets):
            if np.abs(energy[istate]-energy[jstate]) <= threshold_dE:
                for itriplet in range(3):
                    magnitude = np.sqrt(hopping_magnitude[istate][jstate])/np.sqrt(np.abs(energy[istate]-energy[jstate]))/2
                    magnitude2= np.sqrt(hopping_magnitude2[istate][jstate]**2/(np.abs(energy[istate]-energy[jstate])**2))/2
                    nonadiabatic_couplings[istate+n_triplets*itriplet][jstate+n_triplets*itriplet]=hopping_direction[istate][jstate] * (magnitude+magnitude2)
                    nonadiabatic_couplings[istate+n_triplets*itriplet][jstate+n_triplets*itriplet] = -nonadiabatic_couplings[jstate+n_triplets*itriplet][istate+n_triplets*itriplet]
    for istate in range(n_states):
        for jstate in range(n_states):
            g_string+="State %d %d\n" %(istate+1,jstate+1)
            for iatom in range(n_atoms):
                for xyz in range(3):
                    g_string += "%20.12E " %(nonadiabatic_couplings[istate][jstate][iatom][xyz])
                g_string += "\n"
    return g_string
def get_nacs_deltaH(hessian,energy,forces,n_states):
    n_singlets=n_states['n_singlets']
    n_triplets=n_states['n_triplets']
    n_states = n_states['n_states']
    n_atoms = forces.shape[1]
    g_string="! 5 Non-adiabatic Couplings (ddr) (%dx%dx%dx3, real) \n" % (n_states, n_states, n_atoms)
    nonadiabatic_couplings = np.zeros((n_states,n_states,n_atoms,3))
    iterator = -1
    forces = -forces
    hopping_direction=np.zeros( (n_singlets + n_triplets,n_singlets + n_triplets, n_atoms, 3) )
    hopping_magnitude= np.zeros( (n_singlets + n_triplets, n_singlets + n_triplets,1) )
    for istate in range(n_singlets):
        for jstate in range(istate+1,n_singlets):
            Hi=hessian[istate]
            dE=(energy[istate]-energy[jstate])
            Hj=hessian[jstate]
            GiGi=np.dot(forces[istate].reshape(-1,1),forces[istate].reshape(-1,1).T)
            GjGj=np.dot(forces[jstate].reshape(-1,1),forces[jstate].reshape(-1,1).T)
            GiGj=np.dot(forces[istate].reshape(-1,1),forces[jstate].reshape(-1,1).T)
            GjGi=np.dot(forces[jstate].reshape(-1,1),forces[istate].reshape(-1,1).T)
            G_diff = 0.5*(forces[istate]-forces[jstate])
            G_diff2= np.dot(G_diff.reshape(-1,1),G_diff.reshape(-1,1).T)
            dH_2_ij = 0.5*(dE*(Hi-Hj) + GiGi + GjGj - 2*GiGj)
            magnitude = dH_2_ij/2-G_diff2

            #SVD
            u,s,vh=np.linalg.svd(magnitude)
            ev=vh[0]
            ew=s[0]
            iterator=-1

            for iatom in range(n_atoms):
                for xyz in range(3):
                    iterator+=1
                    hopping_direction[istate][jstate][iatom][xyz] = ev[iterator]
                    hopping_direction[jstate][istate][iatom][xyz] = -ev[iterator]
            hopping_magnitude[istate][jstate]=ew
            hopping_magnitude[jstate][istate]=ew

    for istate in range(n_singlets,n_singlets+n_triplets):
        for jstate in range(istate+1,n_singlets+n_triplets):
            Hi=hessian[istate]
            dE=np.abs(energy[istate]-energy[jstate])
            Hj=hessian[jstate]
            GiGi=np.dot(forces[istate].reshape(-1,1),forces[istate].reshape(-1,1).T)
            GjGj=np.dot(forces[jstate].reshape(-1,1),forces[jstate].reshape(-1,1).T)
            GiGj=np.dot(forces[istate].reshape(-1,1),forces[jstate].reshape(-1,1).T)
            GjGi=np.dot(forces[jstate].reshape(-1,1),forces[istate].reshape(-1,1).T)
            G_diff = 0.5*(forces[istate]-forces[jstate])
            G_diff2= np.dot(G_diff.reshape(-1,1),G_diff.reshape(-1,1).T)
            dH_2_ij = 0.5*(dE*(Hi-Hj) + GiGi + GjGj - 2*GiGj)
            magnitude = dH_2_ij/2-G_diff2
            iterator=-1
            u,s,vh = np.linalg.svd(magnitude)
            ev=vh[0]
            ew=s[0]
            iterator=-1
            for iatom in range(n_atoms):
                for xyz in range(3):
                    iterator+=1
                    hopping_direction[istate][jstate][iatom][xyz] = ev[iterator]
                    hopping_direction[jstate][istate][iatom][xyz] = -ev[iterator]
            hopping_magnitude[istate][jstate]=ew
            hopping_magnitude[jstate][istate]=ew
    threshold_dE = 0.04

    compute_nac=False
    for istate in range(n_singlets):
        for jstate in range(istate+1, n_singlets):
            if np.abs(energy[istate]-energy[jstate]) <= threshold_dE:
                magnitude =(hopping_magnitude[istate][jstate])/((np.abs(energy[istate]-energy[jstate])))/2
                nonadiabatic_couplings[istate][jstate][:][:] = hopping_direction[istate][jstate] * magnitude
                nonadiabatic_couplings[jstate][istate][:][:] = -nonadiabatic_couplings[istate][jstate]
    for istate in range(n_singlets,n_singlets+n_triplets):
        for jstate in range(istate+1,n_singlets+n_triplets):
            if np.abs(energy[istate]-energy[jstate]) <= threshold_dE:
                for itriplet in range(3):
                    magnitude = (hopping_magnitude[istate][jstate])/(np.abs(energy[istate]-energy[jstate]))/2
                    nonadiabatic_couplings[istate+n_triplets*itriplet][jstate+n_triplets*itriplet]=hopping_direction[istate][jstate] * magnitude
                    nonadiabatic_couplings[istate+n_triplets*itriplet][jstate+n_triplets*itriplet] = -nonadiabatic_couplings[jstate+n_triplets*itriplet][istate+n_triplets*itriplet]
    for istate in range(n_states):
        for jstate in range(n_states):
            g_string+="State %d %d\n" %(istate+1,jstate+1)
            for iatom in range(n_atoms):
                for xyz in range(3):
                    g_string += "%20.12E " %(nonadiabatic_couplings[istate][jstate][iatom][xyz])
                g_string += "\n"
    return g_string
    
def get_nacs(nacs,n_states):
    n_singlets=n_states['n_singlets']
    n_triplets=n_states['n_triplets']
    n_states = n_states['n_states']
    n_atoms = nacs.shape[1]
    g_string="! 5 Non-adiabatic Couplings (ddr) (%dx%dx%dx3, real) \n" % (n_states, n_states, n_atoms)
    nonadiabatic_couplings = np.zeros((n_states,n_states,n_atoms,3))
    iterator = -1
    for istate in range(n_singlets):
        for jstate in range(istate+1,n_singlets):
            iterator+=1
            nonadiabatic_couplings[istate][jstate] = nacs[iterator]
            nonadiabatic_couplings[jstate][istate] = -nacs[iterator]
    for istate in range(n_singlets,n_singlets+n_triplets):
        for jstate in range(istate+1,n_singlets+n_triplets):
            iterator+=1
            nonadiabatic_couplings[istate][jstate] = nacs[iterator]
            nonadiabatic_couplings[jstate][istate] = -nacs[iterator]
    for istate in range(n_singlets+n_triplets):
        for jstate in range(n_singlets+n_triplets):
            g_string+="State %d %d\n" %(istate+1,jstate+1)
            for iatom in range(n_atoms):
                for xyz in range(3):
                    g_string += "%20.12E " %(nonadiabatic_couplings[istate][jstate][iatom][xyz])
                g_string += "\n"
    return g_string
