#!/usr/remote/bin/python -u
import numpy as np
import os
from sys import argv
import copy
import math
import re
import stat
import shutil
from optparse import OptionParser
import readline

def read_output_dat(sharc_dat,threshold_S,threshold_T):
    """ used class: output_dat
    """
    dict_properties = { "Step"              : False,
                        "Energy"            : False,
                        "SpinOrbitCoupling"	: False,
                        "Dipole"            : False,
                        "NonAdiabaticCoupling"	: False,
			"Gradient"		: False,
			"Index"			: False }
    y=output_dat(sharc_dat)
    ezero = y.ezero
    all_atype = y.all_atype
    nmstates = y.nmstates
    natoms = y.natoms
    NAC=True #NAC=False
    n_singlets = y.states[0]
    n_triplets = y.states[2]
    n_states = n_singlets+3*n_triplets
    stepsize = len(y.startlines)
    all_energy=[]
    #for socs there are n_singlets*n_triplets values for a,b and c and 0.5*(n_triplets*(n_triplets-1)) values for d,e and f
    soc_numberoftriangular=(n_states*(n_states-1))/2
    soc_numberoftriangular=int(soc_numberoftriangular)
    #complex
    all_soc=np.zeros((stepsize,soc_numberoftriangular*2))
    all_soc_real=[]
    all_soc_imag=[]
    all_soc_int=[]
    all_dipole_x=[]
    all_dipole_y=[]
    all_dipole_z=[]
    all_grad = []
    all_geom = []
    all_nac = []
    all_step=np.arange(stepsize)
    dipole_numberofmatrix=(n_states*(n_states-1))/2+n_states #3*((n_singlets*(n_singlets-1)/2+n_singlets)+(n_triplets*(n_triplets-1)/2+n_triplets))
    dipole_numberofmatrix=int(dipole_numberofmatrix)
    all_dipole=np.zeros((stepsize,dipole_numberofmatrix))
    #iterates over steps in output.dat file, every step is a new row in the desired matrices for the quantities: all_quantity
    #dict_properties["Index"]=indices
    #f=open("socfile","w")
    is_ = 0
    it = 0
    for step,read_dict in y:
      #step arrays are overwritten after every iteration over one step and written to the matrices all_quantity
      energy_step=[]
      soc_step = []
      dipole_step_x = []
      dipole_step_y = []
      dipole_step_z = []
      grad_step = []
      nac_step = []
      for singlets in range(n_states):
        energy_step.append(read_dict['Hamiltonian_real'][singlets][singlets])
      SKIP = False
      for istate in range(n_singlets):
        for jstate in range(istate+1,n_singlets):
            deltaE = np.abs(energy_step[istate] - energy_step[jstate])
            if deltaE <= threshold_S:
              SKIP = True
              is_ += 1
      for istate in range(n_singlets,n_singlets+n_triplets):
        for jstate in range(istate+1,n_singlets+n_triplets):
            deltaE = np.abs(energy_step[istate] - energy_step[jstate])
            if deltaE <= threshold_T:
              SKIP = True
              it += 1
      #if SKIP == True:
      #  print(SKIP)
      soc_formean=[]
      if SKIP == False:
        all_geom.append(read_dict["Geometries"])
        all_energy.append(energy_step)
 
        #get the dipole moments 
        for istate in range(n_states):
          for jstate in range(istate,n_states):
            dipole_step_x.append(read_dict['Dipole_x'][istate][jstate])
            dipole_step_y.append(read_dict['Dipole_y'][istate][jstate])
            dipole_step_z.append(read_dict['Dipole_z'][istate][jstate])
        all_dipole_x.append(dipole_step_x)
        all_dipole_y.append(dipole_step_y)
        all_dipole_z.append(dipole_step_z)

        #get socs
        for istate in range(n_states):
            for jstate in range(istate+1,n_states):
               soc_step.append(read_dict['Hamiltonian_real'][istate][jstate])
               soc_step.append(read_dict['Hamiltonian_imag'][istate][jstate])
               #f.write("%s\n" %read_dict['Hamiltonian_real'][istate][jstate])
               #f.write("%s\n" %read_dict['Hamiltonian_imag'][istate][jstate])
               soc_formean.append(read_dict['Hamiltonian_real'][istate][jstate])
               soc_formean.append(read_dict['Hamiltonian_imag'][istate][jstate])

        all_soc_int.append(soc_step)

        for grad_state in range(natoms*n_states):
          for g in range(3):
            grad_step.append(read_dict['Gradient'][grad_state][g])
        all_grad.append(grad_step)
        #get all singlet-singlet nacs without the first (S0 with S0 - this is 0)
        #if those are extracted, get all Singlet couplings with the S1
        #couplings for triplets with different ms are the same, except for ms=+1 - it differs in sign
        if NAC==True:
          singlet=0
          while True:
            if singlet<n_singlets:
              for nac_state_singlet in range(natoms*(singlet+1),natoms*n_singlets):
                for n in range(3):
                  nac_step.append(read_dict['NonAdiabaticCoupling'][nac_state_singlet+singlet*natoms*nmstates][n])
              singlet+=1
            else:
              break
          triplet=0
          while True:
            if triplet<n_triplets:
              #write couplings with ms=-1
              for nac_state_triplet1 in range(natoms*nmstates*(singlet)+natoms*(singlet+triplet+1),natoms*(singlet+n_triplets)+natoms*(singlet)*nmstates):
                for n in range(3):
                  nac_step.append(read_dict['NonAdiabaticCoupling'][nac_state_triplet1+triplet*natoms*nmstates][n])
              triplet+=1
            else:
              break
          all_nac.append(nac_step)
      else:
        pass
    #transform list into matrix, for all_energy: add zero point energy 
    #f.close()
    soc_formean=np.array(soc_formean)
    mean=np.mean(soc_formean)
    std=np.std(soc_formean)
    mini=np.min(soc_formean)
    maxi=np.max(soc_formean)
    print(is_,"Singlets egap < ",threshold_S)
    print(it,"Triplets egap < ",threshold_T)
    all_dipole_x=np.array(all_dipole_x)
    all_dipole_y=np.array(all_dipole_y)
    all_dipole_z=np.array(all_dipole_z)
    all_energy = np.array(all_energy)
    all_energy = all_energy + ezero
    all_soc = np.array(all_soc_int)
    all_grad = np.array(all_grad)
    all_nac = np.array(all_nac)
    all_geom = np.array(all_geom)
    stepsize=all_geom.shape[0]
    all_dipole=np.zeros((stepsize,3*all_dipole_x.shape[1]))
    for t in range(stepsize):
      for x in range(all_dipole_x.shape[1]):
        all_dipole[t][x]=all_dipole_x[t][x]
        all_dipole[t][x+all_dipole_x.shape[1]]=all_dipole_y[t][x]
        all_dipole[t][x+2*all_dipole_x.shape[1]]=all_dipole_z[t][x]
    #conversion of geometry matrix with values in a.u. to values given in angstrom
    all_geom = all_geom*0.529177211
    tensor_dimension = 1+all_energy.shape[1]+all_soc.shape[1]+all_dipole.shape[1]+all_grad.shape[1] #+all_nac.shape[1]
    dict_properties["AllAtomTypes"]=all_atype
    dict_properties["AllGeometries"]=all_geom
    dict_properties["NumberOfAtoms"]=natoms
    dict_properties["Step"]=all_step
    dict_properties["Stepsize"]=stepsize
    dict_properties["Tensordimension"]=tensor_dimension
    dict_properties["Energy"]=all_energy
    dict_properties["SpinOrbitCoupling"]=all_soc
    dict_properties["Dipole"]=all_dipole
    dict_properties["NonAdiabaticCoupling"]=all_nac
    dict_properties["Gradient"]=all_grad
    dict_properties["NumberOfStates"]=read_dict["NumberOfStates"]
    dict_properties["Ezero"]=ezero
    dict_properties["n_Singlets"]=n_singlets
    dict_properties['n_Triplets']=n_triplets
    return dict_properties


def readfile(sharc_dat):
    try:
      f=open(sharc_dat).readlines()
    except IOError as e:
      exit()
    return f 

class output_dat:
  def __init__(self,sharc_dat):
    #self._iter=iter(sharc_dat)
    self.data=readfile(sharc_dat)
    self.sharc_dat = sharc_dat
    # get atom types
    # get number of atoms
    #get line numbers where new timesteps start
    self.startlines=[]
    self.startline=[]
    iline=-1
    jline=-1
    while True:
      iline+=1
      jline+=1
      if iline==len(self.data):
        break
      if jline==len(self.data):
        break
      if 'Step' in self.data[iline]:
        self.startlines.append(iline)
      if '! Elements' in self.data[jline]:
        self.startline.append(jline)
    self.current=0
    for line in self.data:
      #line = line.strip()
      if 'natom' in line:
        a=line.split()[1]
        #self.natoms= int(a)
        break
    self.natoms= int(a)
    # get number of states
    for line in self.data:
      #line = line.strip()
      if 'nstates_m' in line:
        s=line.split()[1:]
        #self.states=[ int(i) for i in s ]
        break
    self.states=[ int(i) for i in s ]
    self.all_atype=[]
    for r in range(self.natoms):
      index=1+r+self.startline[self.current]
      line=self.data[index]
      t=line.split()
      t = t[0]
      self.all_atype.append(str(t))
    self.current=0
    for line in self.data:
      if 'write_grad' in line:
        gradientindex=line.split()[1]
        self.gradientindex=int(gradientindex)
      if 'write_overlap' in line:
        overlapindex=line.split()[1]
        self.overlapindex=int(overlapindex)
      if 'write_property1d' in line:
        prop1dindex=line.split()[1]
        self.prop1dindex=int(prop1dindex)
      if 'write_property2d' in line:
        prop2dindex=line.split()[1]
        self.prop2dindex=int(prop2dindex)
      if 'write_nacdr' in line:
        nacindex=line.split()[1]
        self.nacindex=int(nacindex)
      if 'ezero' in line:
        t=line.split()[1]
        self.ezero = float(t)
      if 'n_property1d' in line:
        n_1dindex=line.split()[1]
        self.n_1dindex=int(n_1dindex)
        if self.prop1dindex==0:
          self.n_1dindex=int(0)
        else:
          pass
      if 'n_property2d' in line:
        n_2dindex=line.split()[1]
        self.n_2dindex=int(n_2dindex)
        if self.prop2dindex==0:
          self.n_2dindex=int(0)
        else:
          pass
        break
      nm=0
      for i,n in enumerate(self.states):
        nm+=n*(i+1)
      self.nmstates=nm

  def __iter__(self):
    return self

  def __next__(self):
    # returns time step, U matrix and diagonal state
    # step
    self.read_dict = {  "Hamiltonian_real"  : False,
                          "Hamiltonian_imag"  : False,
                          "Dipole_x"			: False,
                          "Dipole_y"			: False,
                          "Dipole_z"			: False,
                          "Gradient"			: False,
                          "NonAdiabaticCoupling"		: False }
    read_dict=self.read_dict
    current=self.current
    self.current+=1
    if current+1>len(self.startlines):
      raise StopIteration
    # get rid of blank lines and comment
    # generate lists
    # set counter to zero
    # parse data
    # based on size initialize all relevant arrays

    #Geometries
    all_geom = [ [ 0 for i in range(3) ] for j in range(self.natoms) ]
    for iline in range(self.natoms):
      if self.overlapindex == 0:
        index=self.startlines[current]+18+7*self.nmstates+iline
      else:
        index=self.startlines[current]+19+8*self.nmstates+iline
      line=self.data[index]
      s=line.split()
      for j in range(3):
        all_geom[iline][j]=float(s[j])
    self.read_dict["Geometries"]=all_geom
    #Hamiltonian (Energies)
    H_real=[ [ 0 for i in range(self.nmstates) ] for j in range(self.nmstates) ]
    H_imag=[ [ 0 for i in range(self.nmstates) ] for j in range(self.nmstates) ]
    for iline in range(self.nmstates):
      index=self.startlines[current]+3+iline
      line=self.data[index]
      s=line.split()
      for j in range(self.nmstates):
        H_real[iline][j]=float(s[2*j])
        H_imag[iline][j]=float(s[2*j+1])
    self.read_dict["Hamiltonian_real"]=H_real
    self.read_dict["Hamiltonian_imag"]=H_imag
    self.read_dict["NumberOfStates"]=len(H_real)
    
    Dipole_x=[ [ 0 for i in range(self.nmstates) ] for j in range(self.nmstates) ]
    for iline in range(self.nmstates):
      index=self.startlines[current]+5+2*self.nmstates+iline
      line=self.data[index]
      s=line.split()
      for j in range(self.nmstates):
        Dipole_x[iline][j]=float(s[2*j])
    self.read_dict["Dipole_x"]=Dipole_x
    Dipole_y=[ [ 0 for i in range(self.nmstates) ] for j in range(self.nmstates) ]
    for iline in range(self.nmstates):
      index=self.startlines[current]+6+3*self.nmstates+iline
      line=self.data[index]
      s=line.split()
      for j in range(self.nmstates):
        Dipole_y[iline][j]=float(s[2*j])
    self.read_dict["Dipole_y"]=Dipole_y
    Dipole_z=[ [ 0 for i in range(self.nmstates) ] for j in range(self.nmstates) ]
    for iline in range(self.nmstates):
      index=self.startlines[current]+7+4*self.nmstates+iline
      line=self.data[index]
      s=line.split()
      for j in range(self.nmstates):
        Dipole_z[iline][j]=float(s[2*j])
    self.read_dict["Dipole_z"]=Dipole_z
    
    Gradient=[ [0 for i in range(3) ] for j in range(self.natoms*self.nmstates) ]
    if self.gradientindex==1:
      for iline in range(self.nmstates):
        if self.overlapindex==0:
          index=self.startlines[current]+19+self.n_1dindex+self.n_2dindex+2*self.natoms+(7+self.n_1dindex+self.n_2dindex)*self.nmstates+(iline)*(self.natoms+1)+1
        else:
          index=self.startlines[current]+20+self.n_1dindex+self.n_2dindex+2*self.natoms+(8+self.n_1dindex+self.n_2dindex)*self.nmstates+(iline)*(self.natoms+1)+1
        for a in range(self.natoms):
          line=self.data[index+a]
          s=line.split()
          for j in range(3):
            l=self.natoms*iline+a
            Gradient[l][j]=float(s[j])
    self.read_dict["Gradient"]=Gradient
    
    Nac=[ [0 for i in range(3) ] for j in range(self.natoms*self.nmstates*self.nmstates) ]
    if self.nacindex == 1:
      for iline in range(self.nmstates*self.nmstates):
        if self.overlapindex==0 and self.gradientindex==0:
          index=self.startlines[current]+19+self.n_1dindex+self.n_2dindex+2*self.natoms+(7+self.n_1dindex+self.n_2dindex)*self.nmstates+(iline)*(self.natoms+1)+1
        elif self.overlapindex==1 and self.gradientindex==0:
          index=self.startlines[current]+20+self.n_1dindex+self.n_2dindex+2*self.natoms+(8+self.n_1dindex+self.n_2dindex)*self.nmstates+(iline)*(self.natoms+1)+1
        elif self.gradientindex==1 and self.overlapindex==0:
          index =self.startlines[current]+19+self.n_1dindex+self.n_2dindex+self.natoms*self.nmstates+2*self.natoms+(8+self.n_1dindex+self.n_2dindex)*self.nmstates+(iline)*(self.natoms+1)+1
        elif self.gradientindex==1 and self.overlapindex==1:
          index =self.startlines[current]+20+self.n_1dindex+self.n_2dindex+self.natoms*self.nmstates+2*self.natoms+(9+self.n_1dindex+self.n_2dindex)*self.nmstates+(iline)*(self.natoms+1)+1
        for a in range(self.natoms):
          line=self.data[index+a]
          s=line.split()
          for j in range(3):
            l=self.natoms*iline+a
            Nac[l][j]=float(s[j])
    self.read_dict["NonAdiabaticCoupling"]=Nac
    
    return current, read_dict

#========================================================================================================

def gen_QMin(INFOS,iconddir,icond):
  #os.system( "printf '' > QM.in" )
  copydir=INFOS['copyQMin']
  QMin=open('%s/%sQM/QM.in' %(copydir,iconddir),'w')
  initfile=INFOS['initcondsexcited']
  initconds=open(initfile,'r').readlines()
  #initfile=open(initconds,'r').readlines()
  startline=[]
  states=INFOS['states']
  natoms=INFOS['natom']
  string='%s\n\n' %(natoms)
  iline=-1
  current=0
  while True:
    iline+=1
    line=initconds[iline]
    if iline>=len(initconds):
      string+='end'
      break
    if 'Index' in initconds[iline]:
      startline.append(iline)
      current+=1
      #set current+1 if Index is in the file initconds - if later startline[current] is chosen: the line starts at the current's time Index was read in the initconds file
      if current>icond:
        break
      if 'Index' in line and '%s' %(icond) in line:
        for iline in range(natoms):
          iline+=1
          index=startline[current-1]+1+iline
          line=initconds[index]
          t=line.split()
          #string+='%s %s %s %s\n' %(t[0], t[2], t[3], t[4]) these values are given in bohr - but they are needed in angstrom
          string+='%s %s %s %s\n' %(t[0], float(t[2])*0.529177211, float(t[3])*0.529177211, float(t[4])*0.529177211)
  string+='unit angstrom\n'
  string+='states '
  for i in range(len(states)):
    string+='%s ' %(states[i])
  string+='\ndt 0\nsavedir %s/%s\n' %(copydir,iconddir)
  string+='SOC\nDM\nGrad all\nNACdr\nphases'
  QMin.write(string)
  QMin.close()
  #copy QM.in file to the necessary path, i.e. the State_x/TRAJ_xxxxx/QM folder
  #cpfrom='QM.in' %(copydir,iconddir)
  #cpto='%s/%sQM/QM.in' %(copydir,iconddir)
  #shutil.copy(cpfrom,cpto)

def get_xyz(dict_properties):
  #iteratre over number of geometries and write a separate file
  stepsize=(dict_properties['AllGeometries'].shape[0])
  for i in range(stepsize):
    file=open("%07d.xyz"%(i+1), "w")
    file.write("%i \n"%dict_properties['NumberOfAtoms'])
    file.write("Charge = +1\n" )
    for j in range((dict_properties['NumberOfAtoms'])):
      atomtype=dict_properties['AllAtomTypes'][j]
      file.write("%s %12.9E %12.9E %12.9E \n"%(atomtype,dict_properties['AllGeometries'][i][j][0],dict_properties['AllGeometries'][i][j][1], dict_properties['AllGeometries'][i][j][2]))
    file.close()

def get_properties(dict_properties):
  #iterate over number of geometries and write a file containing all the properties
  stepsize = dict_properties['Energy'].shape[0]
  NAC=True
  SOC=True
  for i in range(stepsize):
    file = open ("%07d"%(i+1),"w")
    file.write("Singlets %i\nTriplets %i \nEnergy 1\nDipole 1\nSOC 1\nGrad 1\nNAC 1\n"%(dict_properties['n_Singlets'],dict_properties['n_Triplets']))
    file.write("\n! Energy %i\n"%len(dict_properties['Energy'][i]))
    for ener in range(dict_properties['NumberOfStates']):
      file.write("%12.9E "%dict_properties['Energy'][i][ener] )
    if SOC==True:
      file.write("\n! SpinOrbitCoupling %i\n" %len(dict_properties['SpinOrbitCoupling'][i]))
      for soc in range(len(dict_properties['SpinOrbitCoupling'][i])):
        file.write("%12.9E "%dict_properties['SpinOrbitCoupling'][i][soc])
    file.write("\n! Dipole %i\n"%len(dict_properties['Dipole'][i]))
    for dipole in range(len(dict_properties['Dipole'][i])):
      file.write("%12.9E "%dict_properties['Dipole'][i][dipole])
    file.write("\n! Gradient %i\n"%len(dict_properties['Gradient'][i]))
    for grad in range(len(dict_properties['Gradient'][i])):
      file.write("%12.9E "%dict_properties['Gradient'][i][grad])
    if NAC==True:
      file.write("\n! Nonadiabatic coupling %i\n" %(len(dict_properties['NonAdiabaticCoupling'][i])))
      for nac in range(len(dict_properties['NonAdiabaticCoupling'][i])):
        file.write("%12.9E "%dict_properties['NonAdiabaticCoupling'][i][nac])
    file.close()

if  __name__ == "__main__":
    try:
        name, datafile,threshold_S,threshold_T = argv
    except ValueError:
        print( "Usage: script <filename>")
        exit()
    threshold_S = float(argv[2])
    threshold_T = float(argv[3])
    dict_properties = read_output_dat(datafile,threshold_S,threshold_T)
    get_xyz(dict_properties)
    get_properties(dict_properties)
    os.system("mkdir xyz-files properties")
    os.system("mv *.xyz xyz-files")
    os.system("mv 0* properties")

