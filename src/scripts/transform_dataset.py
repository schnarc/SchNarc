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
        doublets = 0
        triplets = 0
        quartets = 0
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
        _dyson = False
        property_matrix=False
        dyson = np.zeros((1))
        property_list=[]
        for line in prop_file:
            if line.startswith("Singlets"):
                singlets = int(line.split()[1])
            elif line.startswith("Doublets"):
                doublets = int(line.split()[1])
            elif line.startswith("Triplets"):
                triplets = int(line.split()[1])
            elif line.startswith("Quartets"):
                quartets = int(line.split()[1])
            elif line.startswith("Energy"):
                if int(line.split()[-1])==int(1):
                    _energy = True
                    property_list.append('energy')
            elif line.startswith("Dipole"):
                if int(line.split()[-1])==int(1):
                    _dipole = True
                    property_list.append('dipoles')
            elif line.startswith("SOC"):
                if int(line.split()[-1])==int(1):
                    _soc = True
                    property_list.append('socs')
            elif line.startswith("Grad"):
                if int(line.split()[-1])==int(1):
                    _force = True
                    property_list.append('forces')
                    property_list.append('has_forces')
            elif line.startswith("Given_grad"):
                has_force=[]
                if int(line.split()[-1])==int(1):
                    _has_forces = True
                    has_force.append(1)
                    property_list.append('has_forces')
                else:
                    has_force.append(0)
                has_force=np.array(has_force)
            elif line.startswith("NAC"):
                if int(line.split()[-1])==int(1):
                    _nac = True
                    property_list.append('nacs')
            elif line.startswith('DYSON'):
                if int(line.split()[-1])==int(1):
                    _dyson = True
                    property_list.append('dyson')
            else:
                continue
        nmstates = singlets + 2*doublets + 3*triplets + 4*quartets
        iline = -1
        for line in prop_file:
            iline+=1
            if line.startswith("! Energy"):
                n_energy = singlets + doublets + triplets + quartets
                #int(line.split()[2])
                energy = [] #np.zeros((n_energy))
                eline  = prop_file[iline+1].split()
                for i in range(singlets):
                    energy.append(float(eline[i]))
                for i in range(singlets,singlets+doublets):
                    energy.append(float(eline[i]))
                for i in range(singlets+2*doublets,singlets+2*doublets+triplets):
                    energy.append(float(eline[i]))
                for i in range(singlets+2*doublets+3*triplets,singlets+2*doublets+3*triplets+quartets):
                    energy.append(float(eline[i]))
                energy=np.array(energy)
            #dipole is read in as mu(1,1), mu(1,2), mu(1,3),...
            elif line.startswith("! Dipole"):
                n_dipole = int((singlets*(singlets+1))/2+(doublets*(doublets+1))/2+(triplets*(triplets+1))/2+(quartets*(quartets+1))/2)
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
                force = np.zeros((singlets+triplets+doublets+quartets,natom,3))
                index = -1
                gline = prop_file[iline+1].split()
                for istate in range(singlets+doublets):
                    for iatom in range(natom):
                        for xyz in range(3):
                            index+=1
                            force[istate][iatom][xyz] = -float(gline[index])
                index+=(natom*3*doublets)
                for istate in range(singlets+doublets,singlets+doublets+triplets):
                    for iatom in range(natom):
                        for xyz in range(3):
                            index+=1
                            force[istate][iatom][xyz] = -float(gline[index])
                index+=(2*natom*3*triplets)
                for istate in range(singlets+doublets+triplets,singlets+doublets+triplets+quartets):
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
            elif line.startswith('! Dyson'):
                n_dyson = int(line.split()[-1])
                property_matrix = []
                sline = prop_file[iline+1].split()
                for i in range(n_dyson):
                    property_matrix.append(float(sline[i]))
                property_matrix=np.array(property_matrix)
            else:
                continue

        available_properties = { 'energy' : energy,
                        'socs'    : soc,
                        'forces'  : force,
                        'has_forces': has_force,
                        'nacs'    : nac,
                        'dipoles' : dipole,
                        'dyson'   : property_matrix }
        #Append list 
        charge_buffer.append(charge)
        atom_buffer.append(atoms)
        property_buffer.append(available_properties)
    #get schnet format
    metadata['n_singlets'] = int(singlets)
    metadata['n_doublets'] = int(doublets)
    metadata['n_triplets'] = int(triplets)
    metadata['n_quartets'] = int(quartets)
    states = ''
    for singlet in range(singlets):
      states += 'S '
    for dublet in range(2*doublets):
      states += 'D '
    for triplet in range(3*triplets):
      states += 'T '
    for quartet in range(4*quartets):
      states += 'Q '
    metadata['states'] = states
    reference = 'QC' # TODO put your method here
    phasecorrected = False
    metadata['phasecorrected'] = phasecorrected
    metadata['ReferenceMethod'] = reference
    spk_data = AtomsData(filename,available_properties=property_list)
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
