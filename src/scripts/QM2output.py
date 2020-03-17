import os
import numpy as np
from glob import iglob
import shutil
import os
import readline
import pprint
import sys
from sys import argv
#from NN_phasecorrection import check_phases

'''This file should interpolate between a starting geometry taken from a QM.in file and an end-geometry (at which NNs broke down), also taken from a QM.in file_end
      - The files should be transfered into zmat-files and linear interpolation between the start- and end-geometry will be carried out
      - Later, the geometries will be written into the form of a QM.in file, the interface will then generate QM.out files - those will be written into an output.dat format and the output_all.dat can be appended later
      - In the end, the phases should be compared between geometries and corrected in the output.dat files. After the last geometry is corrected, the 
        calculation using a QM-interface, should be carried out with corrected phases
'''



#============================================================================================================================================================#
#======================================================================QM.out 2 output.dat===================================================================#
#============================================================================================================================================================#
def qm2outputdat(Properties,outputdatfile):
  #in order to use the generated data, the important quantities for running NNs (Energy, SOC, Dipole, NAC) will be written into an output.dat format
  #therefore, the header of the output_all.dat file will be taken  -  every matrix (e.g. U matrix, property matrix) will be filled with zeros
  #iterate over QM.out files
  #TODO maybe change Overlap to True again - but right now although overlap and write_overlap is given, no overlaps are written by molpro
  OUTPUT['Overlap']=False
  file=open("output_new.dat", 'w')
  nmstates = OUTPUT['nmstates']
  natoms = OUTPUT['natoms']
  string_output=header
  string_output+='! 0 Step\n 0\n'
  string_hamilton='! 1 Hamiltonian (MCH) in a.u.\n'
  string_umatrix='! 2 U matrix\n'
  string_dipole_x='! 3 Dipole moments X (MCH) in a.u.\n'
  string_dipole_y='! 3 Dipole moments Y (MCH) in a.u.\n'
  string_dipole_z='! 3 Dipole moments Z (MCH) in a.u.\n'
  string_overlap=''
  string_coefficient='! 5 Coefficients (diag)\n'
  string_hopping='! 6 Hopping Probabilities\n'
  string_ekin='! 7 Ekin (a.u.)\n0.000000000\n'
  string_states='! 8 states (diag, MCH)\n0.000000000 0.000000000\n'
  string_random='! 9 Random number\n0.000000000\n'
  string_runtime='! 10 Runtime (sec)\n0.000000000\n'
  #this is done because the file has problems to do the last step - thus one step more is done and an empty file will be written and removed later
  #if index<= nsteps-1:
  string_geom='! 11 Geometry in a.u.\n'
  string_velocities='! 12 Velocities in a.u.\n'
  string_property2d=''
  string_property1d=''
  string_gradient=''
  string_nac=''

  #write string for geometires
  for atomnr in range(natoms):
    string_geom+='%20.12f %20.12f %20.12f\n' %(Properties['Geometries'][atomnr][0],Properties['Geometries'][atomnr][1],Properties['Geometries'][atomnr][2])

  if OUTPUT['Overlap']==True:
    string_overlap+='! 4 Overlap matrix (MCH)\n'
    for numberofstates in range(nmstates):
      for real_imag in range(nmstates*2):
        string_overlap+='0.000000000 '
      string_overlap+='\n'

  for numberofstates in range(nmstates):
    string_coefficient+='0.0000000000 0.0000000000\n'
    string_hopping+='0.000000000 0.000000000\n'
    for real_imag in range(nmstates*2):
      #if phase correction applied: Properties should be replaced with phasecorrected properties. Hamiltonian --> Hamiltonian_phasecorrected 
      if (2*numberofstates)==real_imag:
        string_hamilton+='%20.12f ' %(Properties['Hamiltonian'][numberofstates][real_imag]-OUTPUT['ezero'])
      else:
        string_hamilton+='%20.12f ' %(Properties['Hamiltonian'][numberofstates][real_imag])
      string_umatrix+='0.000000000 '
      if OUTPUT['Dipole'] == True:
          string_dipole_x+='%20.12f ' %Properties['Dipole_x'][numberofstates][real_imag]
          string_dipole_y+='%20.12f ' %Properties['Dipole_y'][numberofstates][real_imag]
          string_dipole_z+='%20.12f ' %Properties['Dipole_z'][numberofstates][real_imag]
      if OUTPUT['Dipole'] == False:
          string_dipole_x+='0.00000 ' #%Properties['Dipole_x'][numberofstates][real_imag]
          string_dipole_y+='0.00000 ' #%Properties['Dipole_y'][numberofstates][real_imag]
          string_dipole_z+='0.00000 ' #%Properties['Dipole_z'][numberofstates][real_imag]
    string_umatrix+='\n'
    string_hamilton+='\n'
    string_dipole_x+='\n'
    string_dipole_y+='\n'
    string_dipole_z+='\n'
  for numberofatoms in range(natoms):
    string_velocities+='0.000000000 0.000000000 0.000000000\n'

  if OUTPUT['Gradient']==True:
    for numberofstates in range(nmstates):
      string_gradient+='! 15 Gradients (MCH) State %i\n' %(int(numberofstates)+1)
      xyz=0
      for numberofatoms in range(natoms*3):
        xyz+=1
        string_gradient+='%20.12f ' %Properties['Gradient'][numberofstates][numberofatoms]
        if xyz>=3:
          xyz=0
          string_gradient+='\n'

  if int(OUTPUT['n_property2d'])==1:
    n_property2d=int(OUTPUT['n_property2d'])
    for number in range(n_property2d):
      string_property2d+='! 13 Property matrix (MCH)  %i : N/A\n' %(int(number)+1)
      for numberofstates in range(nmstates):
        for real_imag in range(nmstates*2):
          string_property2d+='%20.12f '%Properties['Dyson'][numberofstates][real_imag]
        string_property2d+='\n'

  if int(OUTPUT['Property1d'])==1:
    n_property1d=int(OUTPUT['n_property1d'])
    for number in range(n_property1d):
      string_property1d+='! 14 Property vector (MCH)  %i : N/A\n' %(int(number)+1)
      for numberofstates in range(nmstates):
        string_property1d+='0.000000000\n'

  if OUTPUT['NACdr']==True:
    state=0
    firststate=1
    for nacindex in range(nmstates*nmstates):
      state+=1
      string_nac+='! 16 NACdr matrix element (MCH)  %i %i\n' %(firststate, state)
      if state >= nmstates:
        state=0
        firststate+=1
      xyz=0
      for numberofatoms_xyz in range(natoms*3):
        xyz+=1
        string_nac+='%20.12f ' %(Properties['NonAdiabaticCouplings'][nacindex][numberofatoms_xyz])
        if xyz>=3:
          xyz=0
          string_nac+='\n'

  string_output+=string_hamilton+string_umatrix+string_dipole_x+string_dipole_y+string_dipole_z+string_overlap+string_coefficient+string_hopping+string_ekin+string_states+string_random+string_runtime+string_geom+string_velocities+string_property2d+string_property1d+string_gradient+string_nac
  file.write(string_output)
  file.close()

  return OUTPUT

"""def generate_outputdat(scanpath, nsteps):
  #merge all output.dat files to one file called output.dat
  outputfile = open(scanpath+"/output.dat", "w")
  header_exists=False
  nsteps=int(nsteps)
  for index in range(1,nsteps):
    infile=open(scanpath+'/output'+str(index)+'.dat', 'r').readlines()
    is_header=True
    for line in infile:
      if not header_exists:
        outputfile.write(line)
      if 'End of header' in line:
        is_header=False
        header_exists=True
        continue
      if not is_header:
        outputfile.write(line)
  outputfile.close()
  for i in range(1,nsteps):
    os.system('rm %s/output%i.dat' %(scanpath,i))"""


#=================================================================GET HEADER OF OUTPUT.DAT===================================================================#
def get_header(outputdatfile):

  OUTPUT={ 'Overlap':      False,
           'Gradient':     False,
           'NACdr':        False,
           'Dipole':       False,
           'Property1d':   0,
           'Property2d':   0,
           'n_property1d': 1,
           'n_property2d': 1}

  data=open(outputdatfile,'r').readlines()
  iline=-1
  header=''
  for line in data:
    iline+=1
    line_string=data[iline]
    line=line.strip()
    header+='%s' %line_string
    if line.startswith('nstates_m'):
      line = line.split()
      n_singlets=int(line[1])
      n_triplets=int(line[3])
      n_dublets=int(line[2])
      n_quartets=int(line[4])
      nmstates=1*int(line[1])+2*int(line[2])+3*int(line[3])+4*int(line[4])
      OUTPUT['nmstates']=int(nmstates)
      OUTPUT['n_singlets']=int(n_singlets)
      OUTPUT['n_dublets']=int(n_dublets)
      OUTPUT['n_triplets']=int(n_triplets)
      OUTPUT['n_quartets']=int(n_quartets)
    elif line.startswith('natom'):
      line = line.split()
      natoms=line[1]
      OUTPUT['natoms']=int(natoms)
    elif line.startswith('write_overlap'):
      line=line.split()
      if int(line[1])==1:
        OUTPUT['Overlap']=True
    elif line.startswith('write_grad'):
      line=line.split()
      if int(line[1])==1:
        OUTPUT['Gradient']=True
    elif line.startswith('write_nacdr'):
      line=line.split()
      if int(line[1])==1:
        OUTPUT['NACdr']=True
    elif line.startswith('write_property1d'):
      line=line.split()
      if int(line[1])==1:
        OUTPUT['Property1d']=1
    elif line.startswith('write_property2d'):
      line=line.split()
      if int(line[1])==1:
        OUTPUT['Property2d']=1
    elif line.startswith('n_property1d'):
      line = line.split()
      OUTPUT['n_property1d']=int(line[1])
    elif line.startswith('n_property2d'):
      line = line.split()
      OUTPUT['n_property2d']=int(line[1])
    elif line.startswith( 'ezero' ):
      line = line.split()
      ezero = float(line[1])
      OUTPUT['ezero']= ezero
    elif line.startswith('Dipoles'):
        line = line.split()
        if int(line[1]) == 1:
            OUTPUT['Dipole'] = True
    else:
      if 'End of header' in line:
        break
      else:
        continue

  return header, OUTPUT

#==============================================================GET GEOMETRY FROM QM.in FILE===================================================================#
def get_geom(natoms,QMin,Properties):
  #get information of the geometry from the interpolated and aligned xyz-files
  data=open(QMin, 'r').readlines()
  iline=0
  """string_geom='! 11 Geometry in a.u.\n'
  for line in data:
    line.strip()
    iline+=1
    if iline==len(data):
      break
    if line.startswith('tmp.pdb'):
      for atom in range(natoms):
        iline+=1
        line = data[iline]
        line = line.split()
        string_geom+='%20.12f %20.12f %20.12f \n' %(float(line[1]),float(line[2]),float(line[3]))"""	
  Geometries=np.zeros((natoms,3))
  for line in data:
    line.strip()
    iline+=1
    if iline==len(data):
      break
    for atom in range(natoms):
      iline+=1
      line=data[iline]
      #print line
      line=line.split()
      #convert from Angstrom to Bohr
      Geometries[atom][0]=float(line[1])/0.529177
      Geometries[atom][1]=float(line[2])/0.529177
      Geometries[atom][2]=float(line[3])/0.529177
    break
  Properties['Geometries'] = Geometries
  return Properties



#============================================================================================================================================================#
#======================================================================PHASECORRECTION=======================================================================#
#============================================================================================================================================================#
def phasecorrection(Properties,oldfilename,filename,newfilename,outputdatfile,inputfile,QMin):
  #corrects the phases of the Hamiltonian (SOCs), NACs and Dipole Moments (and additionally of Overlaps)
  header,OUTPUT=get_header(outputdatfile)
  nmstates=OUTPUT['nmstates']
  natoms = OUTPUT['natoms']
  n_singlets = OUTPUT['n_singlets']
  n_dublets = OUTPUT['n_dublets']
  n_quartets = OUTPUT['n_quartets']
  n_triplets = OUTPUT['n_triplets']
  #print(n_singlets,n_triplets)
  #if n_triplets==None:
  #  n_triplets = int(0)
  Properties,data=read_QMout(Properties,nmstates,n_singlets,n_triplets,filename)
  #get ezero
  Properties=read_input(Properties,inputfile)
  #multiply the phasevector with the previous phasevector
  phasevector_old = Properties['phasevector_original']
  phasevector = Properties['phasevector']
  phasevector = phasevector_old*phasevector
  #print phasevector_old
  #print Properties['phasevector']
  #print phasevector
  Properties.update({'phasevector': phasevector})
  phasevector2=np.zeros((nmstates*2))
  #double each entry of the phasevector to make calculation of real and imaginary values easier
  for numberofstates in range(nmstates):
    phasevector2[numberofstates*2]=phasevector[numberofstates]
    phasevector2[numberofstates*2+1]=phasevector[numberofstates]

  #DO PHASECORRECTION
  QMout=open(newfilename, "w")
  iline=-1
  line_index=-1
  for line in data:
    line = line.strip()
    iline+=1
    line_index+=1
    if iline==len(data):
      break

    elif line.startswith('! 1 Hamiltonian Matrix'):
      #write the next two lines to file
      jline=line_index
      for i in range(2):
        line = data[jline]
        QMout.write(line)
        jline+=1
      #instead of writing the original Hamiltonian, do phasecorrection and write the new Hamiltonian 
      Hamiltonian=Properties['Hamiltonian']
      #do calculation of rows with vector
      Hamiltonian_phasecorrected = Hamiltonian * phasevector2
      #do calculation of columns with vector by calculation of each row with the first entry of the vector
      for column in range(nmstates):
        for row_element in range(nmstates*2):
          Hamiltonian_phasecorrected[column][row_element]=Hamiltonian_phasecorrected[column][row_element]*phasevector[column]
      Properties.update({'Hamiltonian_phasecorrected': Hamiltonian_phasecorrected})
      #write the Hamiltonian to QM.out (next 8 lines)
      string_hamiltonian=''
      for hamilton in range(nmstates):
        for row_elements in range(nmstates*2):
          string_hamiltonian+='%20.12f '%Hamiltonian_phasecorrected[hamilton][row_elements]
        string_hamiltonian+='\n'
      string_hamiltonian+='\n'
      QMout.write(string_hamiltonian)
      #substract ezero from energy values of Hamiltonian for output.dat
      ezero = Properties['ezero']
      Hamiltonian_phasecorrected_ezero=Hamiltonian_phasecorrected
      for numberofstates in range(nmstates):
        for energyvalues in range(nmstates):
          if numberofstates == energyvalues:
            Hamiltonian_phasecorrected_ezero[numberofstates][energyvalues*2]=Hamiltonian_phasecorrected[numberofstates][energyvalues*2]-ezero
      Properties.update({'Hamiltonian_phasecorrected_ezero': Hamiltonian_phasecorrected_ezero})

    elif line.startswith('! 2 Dipole Moment'):
      #write the next line to the file
      jline = line_index
      line=data[jline]
      QMout.write(line)
      #do phasecorrection
      Dipole_x=Properties['Dipole_x']
      Dipole_y=Properties['Dipole_y']
      Dipole_z=Properties['Dipole_z']
      #
      Dipole_x_phasecorrected = []
      Dipole_y_phasecorrected = []
      Dipole_z_phasecorrected = []
      #
      Dipole_x_phasecorrected = Dipole_x * phasevector2
      Dipole_y_phasecorrected = Dipole_y * phasevector2
      Dipole_z_phasecorrected = Dipole_z * phasevector2
      #
      for column in range(nmstates):
        for row_element in range(nmstates*2):
          Dipole_x_phasecorrected[column][row_element]=Dipole_x_phasecorrected[column][row_element]*phasevector[column]
          Dipole_y_phasecorrected[column][row_element]=Dipole_y_phasecorrected[column][row_element]*phasevector[column]
          Dipole_z_phasecorrected[column][row_element]=Dipole_z_phasecorrected[column][row_element]*phasevector[column]
      #
      Properties.update({'Dipole_x_phasecorrected': Dipole_x_phasecorrected})
      Properties.update({'Dipole_y_phasecorrected': Dipole_y_phasecorrected})
      Properties.update({'Dipole_z_phasecorrected': Dipole_z_phasecorrected})
      #
      string_dipole=''
      jline+=1
      line=data[jline]
      if OUTPUT['Dipole'] == True:
          string_dipole+='%s' %line
          for dipole in range(nmstates):
            for row_elements in range(nmstates*2):
              string_dipole+='%20.12f '%Dipole_x_phasecorrected[dipole][row_elements]
            string_dipole+='\n'
            jline+=1
          jline+=1
          line = data[jline]
          string_dipole+='%s' %line
          for dipole in range(nmstates):
            for row_elements in range(nmstates*2):
              string_dipole+='%20.12f '%Dipole_y_phasecorrected[dipole][row_elements]
            string_dipole+='\n'
            jline+=1
          jline+=1
          line = data[jline]
          string_dipole+='%s'%line
          for dipole in range(nmstates):
            for row_elements in range(nmstates*2):
              string_dipole+='%20.12f '%Dipole_z_phasecorrected[dipole][row_elements]
            string_dipole+='\n'
            jline+=1
          #print string_dipole
          QMout.write(string_dipole)

    elif line.startswith('! 3 Gradient Vectors'):
      #only write lines - no phasecorrection for gradients
      #write the next lines containing gradient information to the file to the file
      jline = line_index
      for gradient_index in range((natoms+1)*nmstates+1):
        line=data[jline]
        QMout.write(line)
        jline+=1

    elif line.startswith('! 5 Non-adiabatic couplings'):
      #write first line to the QM.out file
      jline = line_index
      line = data[jline]
      QMout.write(line)

      #do phasecorrection
      NAC=Properties['NonAdiabaticCouplings']
      #the NAC matrix is saved as a matrix containing natoms*3 entries per line (so one line contains the nacs between a specific state and another specific one)
      #each line will be corrected with the phase - first, every line has to be corrected with the first entry of the phasecorrection vector, the second line with the second,...
      NAC_phasecorrected = np.zeros((nmstates*nmstates,natoms*3))

      #correct by multiplication with phasecorrection_vector line by line
      phasecorrection_index_line=0 
      for nacindex_line in range(nmstates*nmstates):
        for nac_state_line in range(natoms*3):
          NAC_phasecorrected[nacindex_line][nac_state_line] = NAC[nacindex_line][nac_state_line] * phasevector[phasecorrection_index_line]
        phasecorrection_index_line+=1
        if phasecorrection_index_line >= nmstates:
          phasecorrection_index_line=0

      #do phasecorrection column by column
      #this means, the first nmstates*line entries have to be multiplicated by the first entry of the phasecorrection vector
      phasecorrection_index_column=0
      iterator=0
      for nacindex_column in range(nmstates*nmstates):
        for nac_state_column in range(natoms*3):
          NAC_phasecorrected[nacindex_column][nac_state_column] = NAC_phasecorrected[nacindex_column][nac_state_column] * phasevector[phasecorrection_index_column]
        iterator+=1
        if iterator >= nmstates:
          iterator=0
          phasecorrection_index_column+=1
      Properties.update({'NAC_phasecorrected': NAC_phasecorrected})

      #make string for writing to QM.out file
      string_nac=''
      for nacstring in range(nmstates*nmstates):
        #contains information about the states of the NACs
        jline+=1
        line=data[jline]
        string_nac+='%s' %line
        jline+=natoms
        xyz_index=0
        for nacline in range(natoms*3):
          xyz_index+=1
          string_nac+='%20.12f ' %NAC_phasecorrected[nacstring][nacline]
          if xyz_index>=3:
            xyz_index=0
            string_nac+='\n'
      QMout.write(string_nac)

    elif line.startswith('! 6 Overlap matrix'):
      #write the next 2 lines to the QM.out file
      jline = line_index
      for index in range(2):
        line = data[jline]
        QMout.write(line)
        jline+=1

      Overlap=Properties['Overlap']
      #get the old phasevector since the overlaps are mutiplied line by line with the current phasevector and column by column with the one of the previous timestep (<S1|S2>)
      phasevector_old=Properties['phasevector_original']
      #first phasecorrection line by line
      Overlap_phasecorrected = Overlap * phasevector2
      #second phasecorrection column by column
      for lineindex in range(nmstates):
        for overlap_index in range(nmstates*2):
          Overlap_phasecorrected[lineindex][overlap_index] = Overlap_phasecorrected[lineindex][overlap_index] * phasevector_old[lineindex] 
      #prepare String
      string_overlap=''
      for overlap in range(nmstates):
        for line_value in range(nmstates*2):
          string_overlap+='%20.12f ' %Overlap_phasecorrected[overlap][line_value]
        string_overlap+='\n'
      string_overlap+='\n'
      QMout.write(string_overlap)

    elif line.startswith('! 7 Phases'):
      #writes the phase vector
      jline = line_index
      for phasestring in range(2):
        line = data[jline]
        QMout.write(line)
        jline+=1
      string_phase=''
      for phases in range(nmstates):
        string_phase+='%20.12f 0.000000000\n' %phasevector[phases]
      QMout.write(string_phase)

    elif line.startswith('! 11 Property'):
        #write dyson norms
        string_property=''
        line=data[jline]
        print(line)
        print("HER")
        QMout.write(line)
        jline+=1
        Property=Properties['Dyson']
        string_property+='%s' %line
        for dyson in range(nmstates):
          for row_elements in range(nmstates*2):
            string_property+='%20.12f '%Property[dyson][row_elements]
          string_property+="\n"
        QMout.write(string_property)
    else:
      pass

  QMout.close()

  Properties=get_geom(natoms,QMin,Properties)

  return Properties, OUTPUT


#======================================================================READ QM.out===================================================================#
#complementary to function "check_phases" from "NN_phasecorrection.py"

def read_QMinit_out(oldfilename,QMin,OUTPUT):
  Properties={}
  #get the phasevector
  readQMout=open(oldfilename,'r')
  data=readQMout.readlines()
  iline=-1
  for line in data:
    iline+=1
    line = line.strip()
    if iline==len(data):
      break
    line=data[1]
    t = line.split()
    nmstates=t[0]
    nmstates=int(nmstates)
    Properties['nmstates']=nmstates
    break
  iline = -1
  for line in data:
    iline += 1
    line = data[iline]
    line = line.strip()
    if iline == len(data):
      break
    elif line.startswith('! 7 Phases'):
      iline+=1
      line=data[iline]
      line=line.split()
      phasevector=np.zeros((nmstates))
      for numberofstates in range(nmstates):
        line = data[iline+1+numberofstates]
        line = line.split()
        #print line
        phasevector[numberofstates]=line[0]
      Properties['phasevector_original']=phasevector
      #print Properties['phasevector_original']
      break
    else:
      #set the phasevector to +1
      phasevector=np.ones((nmstates))
      Properties['phasevector_original']=phasevector
  Properties,data=read_QMout(Properties,nmstates,OUTPUT['n_singlets'], OUTPUT['n_triplets'],oldfilename)
  Properties=get_geom(OUTPUT['natoms'],QMin,Properties)
  return Properties

def read_QMout(Properties,nmstates,n_singlets,n_triplets,filename):
  #iterate over QM.out files
  Properties.update({'phasevector':		False,
			  'Hamiltonian':		False,
			  'Dipole_x':   		False,
			  'Dipole_y':   		False,
			  'Dipole_z':   		False,
			  'Gradient':  			False,
			  'NonAdibaticCouplings':	False,
			  'Overlap':			False})
  #start from 1 because the inital file does not have to be included 
  readQMout=open(filename, "r")
  data=readQMout.readlines()
  readQMout.close()
  overlap_vector_correct=np.zeros((nmstates))
  threshold = float(0.5)
  iline=-1
  #print( n_singlets, n_triplets)
  for line in data:
    line=line.strip()
    iline+=1
    if line.startswith("! 6 Overlap matrix" ):
      iline+=1
      #check phases of singlets only
      for overlaps_singlets in range(n_singlets):
        iline+=1
        line = data[iline]
        line = line.split()
        overlap_singlets = float(line[2*overlaps_singlets])
        #print( overlap)
        if abs(overlap_singlets) < threshold:
          for overlaps_stateswitch in range(n_singlets):
            overlap_switched_state = float(line[2*overlaps_stateswitch])
            #print (overlap_switched_state)
            if abs(overlap_switched_state) > threshold:
              #print("states switched")
              overlap_singlets = overlap_switched_state
              if overlap_singlets < int(0):
                overlap_singlets = int(-1)
              else:
                overlap_singlets = int(1)
        else:
          if overlap_singlets < int(0):
            overlap_singlets = int(-1)
          else:
            overlap_singlets = int(1)
        overlap_vector_correct[overlaps_singlets]=overlap_singlets
      #check phases of triplets - do this three times since triplets are three times degenerated
      triplet_nmstate_0=int(0)
      triplet_nmstate_1=int(0)
      if n_triplets==0:
        pass
      else:
        for triplet_number in range(1):
          triplet_nmstate_0=triplet_nmstate_1
          triplet_nmstate_1=triplet_nmstate_0+1
          #print(triplet_magnetic, "should start with 1")
          for overlaps_triplets in range(n_singlets*triplet_nmstate_0,n_singlets+triplet_nmstate_1*n_triplets):
            iline+=1 
            line = data[iline]
            line = line.split()
            overlap_triplets = float(line[2*overlaps_triplets])
            if abs(overlap_triplets) < threshold:
              for overlaps_stateswitch_triplets in range(n_singlets*triplet_nmstate_0,n_singlets+triplet_nmstate_1*n_triplets):
                overlap_switched_state_triplet = float(line[2*overlaps_stateswitch_triplets])
                if abs(overlap_switched_state_triplet) > threshold:
                  overlap_triplets = overlap_switched_state_triplet
                  if overlap_triplets < int(0):
                    overlap_triplets = int(-1)
                  else:
                    overlap_triplets = int(1)
            else:
              if overlap_triplets < int(0):
                overlap_triplets = int(-1)
              else:
                overlap_triplets = int(1)
            overlap_vector_correct[overlaps_triplets]=overlap_triplets
    #get number of states and write out vector with phases
    """if line.startswith("! 7 Phases"):
      phasevector=np.zeros((nmstates))
      for numberofstates in range(nmstates):
        line=data[iline+2+numberofstates]
        line=line.split()
        phasevector[numberofstates]=line[0]	
      Properties['phasevector']=phasevector
      break"""
  iline=-1
  #print(overlap_vector_correct)
  Properties['phasevector']=overlap_vector_correct
  for line in data:
    line=line.strip()
    iline+=1
    if line.startswith('! 1 Hamiltonian Matrix'):
      Hamiltonian=np.zeros((nmstates,nmstates*2))
      for numberofstates in range(nmstates):
        line=data[iline+2+numberofstates]
        line=line.split()
        for real_imag in range(nmstates*2):
          Hamiltonian[numberofstates][real_imag]=line[real_imag]
      Properties['Hamiltonian']=Hamiltonian
    elif line.startswith('! 2 Dipole Moment Matrices'):
      Dipole_x=np.zeros((nmstates,nmstates*2))
      Dipole_y=np.zeros((nmstates,nmstates*2))
      Dipole_z=np.zeros((nmstates,nmstates*2))
      for numberofstates in range(nmstates):
        line_x=data[iline+2+numberofstates]
        line_x=line_x.split()
        line_y=data[iline+3+nmstates+numberofstates]
        line_y=line_y.split()
        line_z=data[iline+4+nmstates*2+numberofstates]
        line_z=line_z.split()
        for real_imag in range(nmstates*2): 
          Dipole_x[numberofstates][real_imag]=line_x[real_imag]
          Dipole_y[numberofstates][real_imag]=line_y[real_imag]
          Dipole_z[numberofstates][real_imag]=line_z[real_imag]
      Properties['Dipole_x']=Dipole_x
      Properties['Dipole_y']=Dipole_y
      Properties['Dipole_z']=Dipole_z
    elif line.startswith('! 3 Gradient Vectors'):
      line=data[iline+1]
      line=line.split()
      natoms=int(line[0])
      Properties.update({'natoms': natoms})
      #writes the gradient tensor in form of a matrix  -  every natomsx(x,y,z) matrix is written as one vector
      Gradient=np.zeros((nmstates,natoms*3))
      index=0
      for numberofstates in range(nmstates):
        index+=1
        gradientindex=0
        for atom_index in range(natoms):
          index+=1
          line=data[iline+index]
          line=line.split()
          for xyz in range(3):
            Gradient[numberofstates][gradientindex]=line[xyz]
            gradientindex+=1
      Properties['Gradient']=Gradient
    elif line.startswith('! 5 Non-adiabatic couplings'):
      #the nacs will be written as a matrix with the size: (numberofstatesxnumberofstates)x(natoms*3(xyz))
      NAC=np.zeros((nmstates*nmstates,natoms*3))
      index=0
      for numberofstates in range(nmstates*nmstates):
        index+=1
        nacindex=0
        for atom_index in range(natoms):
          index+=1
          line=data[iline+index]
          line=line.split()
          for xyz in range(3):
            NAC[numberofstates][nacindex]=line[xyz]
            nacindex+=1
      Properties['NonAdiabaticCouplings']=NAC
    elif line.startswith('! 6 Overlap matrix'):
      overlap=np.zeros((nmstates,nmstates*2))
      for numberofstates in range(nmstates):
        line=data[iline+2+numberofstates]
        line=line.split()
        for real_imag in range(nmstates*2):
          overlap[numberofstates][real_imag]=line[real_imag]
      Properties['Overlap']=overlap

    elif line.startswith('! 11 Property Matrix'):
      dyson=np.zeros((nmstates,nmstates*2))
      for numberofstates in range(nmstates):
        line=data[iline+2+numberofstates]
        line=line.split()
        for real_imag in range(nmstates*2):
          dyson[numberofstates][real_imag]=line[real_imag]
      Properties['Dyson']=dyson

  return Properties, data



def read_input(Properties, inputfile):
  inputf=open(inputfile, 'r').readlines()
  for line in inputf:
    line=line.strip()
    if line.startswith( 'ezero' ):
      line = line.split()
      ezero = float(line[1])
      Properties.update({'ezero': ezero})
  return Properties



if __name__ == "__main__":
  try:
    name,outputdatforheader, QMin = argv
  except ValueError:
    print( "Usage: script <output.dat file for header information> <file of geometry used for calcualtion of QMout; often QM.in>")
    exit()
  outputdatfile = argv[1]
  filename=str("QM.out")
  inputfile=argv[2]
  header,OUTPUT=get_header(outputdatfile)
  Properties=read_QMinit_out(filename,QMin,OUTPUT)
  #Properties,OUTPUT=phasecorrection(Properties,filename,outputdatfile,inputfile)
  qm2outputdat(Properties,outputdatfile)
