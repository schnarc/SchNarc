import schnetpack as spk
import readline
import os
from sys import argv
import numpy as np
from ase import Atoms
from ase.units import Bohr
from schnetpack.data import AtomsData
def read_dataset(path,numberofgeoms,filename):

    atom_buffer = []
    property_buffer = []
    charge_buffer = []
    metadata = {}
    for geom in range(1,1+numberofgeoms):

        #Geometry and Atomtypes
        xyz_file = open(path+"/xyz-files/%07d.xyz"%geom,"r").readlines()
        charge = int(xyz_file[1].split()[2])
        natom = int(xyz_file[0].split()[0])
        E=[]
        R=np.zeros((natom,3))
        for iatom in range(natom):
            E.append(xyz_file[iatom+2].split()[0])
            for xyz in range(3):
               R[iatom][xyz] = float(xyz_file[iatom+2].split()[1+xyz])/Bohr
        atoms = Atoms(E,R)

        #Properties
        prop_file = open(path+"/properties/%07d"%geom,"r").readlines()
        singlets = 0
        triplets = 0
        _energy = False
        energy = np.zeros((1))
        _soc = False
        soc = np.zeros((1))
        _force = False
        force = np.zeros((1))
        _dipole = False
        dipole = np.zeros((1))
        _nac = False
        nac = np.zeros((1))
        for line in prop_file:
            if line.startswith("Singlets"):
                singlets = int(line.split()[1])
            elif line.startswith("Triplets"):
                triplets = int(line.split()[1])
            elif line.startswith("Energy"):
                _energy = True
            elif line.startswith("Dipole"):
                _dipole = True
            elif line.startswith("SOC"):
                _soc = True
            elif line.startswith("Grad"):
                _force = True
            elif line.startswith("NAC"):
                _nac = True
            else:
                continue
        nmstates = singlets + 3*triplets
        iline = -1
        for line in prop_file:
            iline+=1
            if line.startswith("! Energy"):
                n_energy = singlets+triplets 
                #int(line.split()[2])
                energy = [] #np.zeros((n_energy))
                eline  = prop_file[iline+1].split()
                for i in range(n_energy):
                    energy.append(float(eline[i]))
                energy=np.array(energy)
            #dipole is read in as mu(1,1), mu(1,2), mu(1,3),...
            elif line.startswith("! Dipole"):
                n_dipole = int(nmstates*(nmstates-1)/2+nmstates)
                dipole = np.zeros((n_dipole,3))
                dline = prop_file[iline+1].split()
                for i in range(n_dipole):
                    for xyz in range(3):
                        dipole[i][xyz] = float(dline[i+n_dipole*xyz])
            elif line.startswith("! SpinOrbitCoupling"):
                n_soc = int(line.split()[2])
                soc = [] #np.zeros((n_soc))
                sline = prop_file[iline+1].split()
                for i in range(n_soc):
                     soc.append(float(sline[i]))
                soc=np.array(soc)
            elif line.startswith("! Gradient"):
                n_grad = int(line.split()[2])
                force = np.zeros((singlets+triplets,natom,3))
                index = -1
                gline = prop_file[iline+1].split()
                for istate in range(singlets+triplets):
                    for iatom in range(natom):
                        for xyz in range(3):
                            index+=1
                            force[istate][iatom][xyz] = -float(gline[index])
            #nonadiabatic couplings are also defined as vectors
            elif line.startswith("! Nonadiabatic coupling"):
                n_nac = int(int(line.split()[3])/3/natom)
                #dimension: nstates(coupled), natoms,xyz(3)
                nac = np.zeros((n_nac,natom,3))
                nacline = prop_file[iline+1].split()
                index=-1
                for i in range(n_nac):
                    for iatom in range(natom):
                        for xyz in range(3):
                            index+=1
                            nac[i][iatom][xyz] = float(nacline[index])
            else:
                continue

        properties = { 'energy' : energy,
                        'socs'    : soc,
                        'forces'  : force,
                        'nacs'    : nac,
                        'dipoles' : dipole }
        #Append list 
        charge_buffer.append(charge)
        atom_buffer.append(atoms)
        property_buffer.append(properties)
    #get schnet format
    metadata['n_singlets'] = int(singlets)
    metadata['n_triplets'] = int(triplets)
    states = ''
    for singlet in range(singlets):
      states += 'S '
    for triplet in range(3*triplets):
      states += 'T '
    metadata['states'] = states
    reference = 'MR-CISD(6,4)/aug-cc-pVDZ, program: COLUMBUS'
    phasecorrected = False
    metadata['phasecorrected'] = phasecorrected
    metadata['ReferenceMethod'] = reference
    spk_data = AtomsData(filename,)
    spk_data.add_systems(atom_buffer,property_buffer)
    #get metadata
    spk_data.set_metadata(metadata)

if __name__ == "__main__":

    try:
        script, filename, natoms, filename = argv
    except IOError:
        print("USAGE: Script.py path_to_trainingset numberofgeometries filename")

#units should be atomic units always!
#forces are -gradients!
path = argv[1]
numberofgeoms = int(argv[2])
filename = str(argv[3])

read_dataset(path,numberofgeoms,filename)
