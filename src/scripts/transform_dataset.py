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

        available_properties = { 'energy' : energy,
                        'socs'    : soc,
                        'forces'  : force,
                        'has_forces': has_force,
                        'nacs'    : nac,
                        'dipoles' : dipole,
                        'dyson'   : False }
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
