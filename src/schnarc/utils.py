from glob import iglob
import shutil
import readline
import pprint
import sys
from sys import argv
import yaml
import logging
import ase.io
import ase
import numpy as np
import schnetpack as spk
from ase.units import Bohr
import os
#SHARC="/user/julia/software/SchNarc/sharc/source/../bin/"

from schnarc.data import Properties


def generate_default_tradeoffs(yamlpath):
    tradeoffs = {p: 1.0 for p in Properties.properties}
    save_tradeoffs(tradeoffs, yamlpath)


def read_tradeoffs(yamlpath):
    with open(yamlpath, 'r') as tf:
        tradeoffs = yaml.load(tf,Loader=yaml.Loader)
    
    logging.info('Read loss tradeoffs from {:s}.'.format(yamlpath))
    return tradeoffs


def save_tradeoffs(tradeoffs, yamlpath):
    with open(yamlpath, 'w') as tf:
        yaml.dump(tradeoffs, tf, default_flow_style=False)

    logging.info('Default tradeoffs written to {:s}.'.format(yamlpath))

def QMout(prediction,modelpath):
    """
    returns predictions in QM.out format useable with SHARC
    """
    QMout_string=''
    QMout_energy=''
    QMout_force=''
    QMout_dipoles=''
    QMout_nacs=''
    if int(prediction['energy'].shape[0]) == int(1):
        for i,property in enumerate(prediction.keys()):
            if property == "energy":
                QMout_energy=get_energy(prediction['energy'][0])
            elif property == "force":
                QMout_force=get_force(prediction['force'][0])
            elif property == "dipoles":
                QMout_dipoles+=get_dipoles(prediction['dipoles'][0],prediction['energy'][0].shape[0])
            elif property == "nacs":
                QMout_nacs=get_nacs(prediction['nacs'][0],prediction['energy'][0].shape[0])
        QM_out = open("%s/QM.out" %modelpath, "w")
        QMout_string=QMout_energy+QMout_dipoles+QMout_force+QMout_nacs
        QM_out.write(QMout_string)
        QM_out.close()

    else:
        for index in range(prediction['energy'].shape[0]):
            os.system('mkdir %s/Geom_%04d' %(modelpath,index+1))
            for i,property in enumerate(prediction.keys()):
                if property == "energy":
                    QMout_energy=get_energy(prediction['energy'][index])
                elif property == "force":
                    QMout_force=get_force(prediction['force'][index])
                elif property == "dipoles":
                    QMout_dipoles+=get_dipoles(prediction['dipoles'][index],prediction['energy'][index].shape[0])
                elif property == "nacs":
                    QMout_nacs=get_nacs(prediction['nacs'][index],prediction['energy'][index].shape[0])
            QM_out = open("QM.out", "w")
            QMout_string=QMout_energy+QMout_dipoles+QMout_force+QMout_nacs
            QM_out.write(QMout_string)
            QM_out.close()
            os.system("mv QM.out %s/Geom_%04d/" %(modelpath,index+1))




def read_QMout(filename,natoms,soc_flag,nsinglets,ntriplets,threshold):
        data={}
        atoms=[]
        
        #number of nacs
        nnacs = int(nsinglets*(nsinglets-1)/2) + int(ntriplets*(ntriplets-1)/2)
        #number of dipole moment values
        ndipoles = int(nsinglets+ntriplets+nnacs)
        #nstates
        nstates = nsinglets+3*ntriplets
        try:
            file = open(filename, "r").readlines()
            skip=False
        except IOError:
            skip=True
        if skip == False:
            found_overlap=False
            #each data point gets one entry in the dictionary
            data={}
            #each data point contains forces, so we set the following key to 1:
            data["has_forces"]=np.array([1]).reshape(-1)
            
            #this iterator will tell us the line number we are in
            iterator=-1
    
            #iterate over all lines in the output
            for line in file:
                iterator+=1
                # Getting energies
                if line.startswith("! 1 Hamiltonian Matrix"):
                    energies = np.zeros((nsinglets+ntriplets,1))
                    for istate in range(nsinglets+ntriplets):
                        energyline = file[iterator+2+istate].split()
                        energies[istate]=float(energyline[2*istate])
                    data["energy"]=energies.reshape(-1)
                    
                    #get spin-orbit couplings
                    if soc_flag == True:
                        #triangular minus diagonal (times two due to complex values)
                        socs = np.zeros((int(nstates*nstates-nstates)))
                        sociterator = -1
                        for istate in range(nstates*2):
                            socsline = file[iterator+2+istate].split()
                            
                            for jstate in range(2+istate*2,2*nstates):
                                sociterator+=1
                                socs[sociterator]=float(socsline[jstate])
                        data["socs"] = socs
                    
                # Getting forces
                if line.startswith("! 3 Gradient Vectors "):
                    forces = np.zeros((nsinglets+ntriplets,natoms,3))
                    for istate in range(nsinglets+ntriplets):
                        for iatom in range(natoms):
                            forceline = file[iterator+2+istate+istate*natoms+iatom].split()
                            for xyz in range(3):
                                forces[istate][iatom][xyz]=float(forceline[xyz])
                    data["forces"]=forces
                
                # Getting nonadiabatic couplings (NACs)
                # Only take the ones that are between different states, so in this case: skip NAC(1,1)=0
                # Note that only one NAC is saved as NAC(1,2)=-NAC(2,1) - numbers in brackets denote states
                if line.startswith("! 5 Non-adiabatic couplings"):
                    nacs = np.zeros((nnacs,natoms,3))
                    naciterator=-1
                    for inac1 in range(nstates):
                        for inac2 in range(nstates):
                            if inac1==inac2  or inac2<inac1 or (inac1 < nsinglets and inac2>=nsinglets) or inac2>=(nsinglets+ntriplets) or inac1 >=(nsinglets+ntriplets):
                                pass
                            else:
                                naciterator+=1
                                for iatom in range(natoms):
                                    nacline = file[iterator+2+inac1*nstates+inac1*natoms*nstates+inac2*natoms+inac2+iatom].split()
                                    for xyz in range(3):
                                        nacs[naciterator][iatom][xyz] = float(nacline[xyz])
                    data["nacs"]=nacs
                
                # Getting values for dipole moments
                # Transition and permanent dipole moments are included in x,y,z direction (last dimension)
                # The values are read line after line starting from the diagonal values
                # In the QM.out file, there are 3 blocks, one block belongs to one direction (x,y,z)
                # SchNarc has ( ndipoles x (xyz) ) as a shape
                # Again we only take one diagonal of the matrix and only real values (imaginary values are 0)
                if line.startswith("! 2 Dipole Moment Matrices "):
                    dipoles = np.zeros((ndipoles,3))
                    for xyz in range(3):
                        dipoleiterator=-1
                        for idipole1 in range(nstates):
                            for idipole2 in range(nstates):
                                if idipole2<idipole1 or (idipole1 < (nsinglets) and idipole2>=nsinglets) or idipole2>=(nsinglets+ntriplets) or idipole1 >=(nsinglets+ntriplets):
                                    pass
                                else:
                                    dipoleiterator+=1
                                    dipoleline = file[iterator+2+xyz+xyz*nstates+idipole1].split()
                                    dipoles[dipoleiterator][xyz] = float(dipoleline[idipole2*2])
                    data["dipoles"]=dipoles
                
                # Read the overlap matrix
                if line.startswith("! 6 Overlap matrix"):
                    phasevector = np.ones((nsinglets+ntriplets))
                    found_overlap = False

                    overlapiterator=-1
                    for istate in range(nsinglets+ntriplets):
                        overlapiterator+=1
                        overlapline = file[iterator+2+overlapiterator].split()
                        #skip imaginary values (all 0)
                        #this asks about the diagonal values of the overlap matrix
                        if np.abs(float(overlapline[2*istate])) >= 0.5:
                            found_overlap=True
                            if float(overlapline[2*istate]) >= 0.5:
                                phasevector[istate] = +1
                            else:
                                phasevector[istate] = -1
                        else:
                            # if the overlap are not large enough for one state, then it could be that states have switched. 
                            # we will check for this below
                            # this asks about all entries
                            for jstate in range(nsinglets+ntriplets):
                                if np.abs(float(overlapline[2*jstate])) >= threshold:
                                    found_overlap = True
                                    if float(overlapline[2*jstate]) >= thresholds:
                                        phasevector[istate] = +1
                                    else:
                                        phasevector[istate] = -1     
                    if found_overlap == True:
                        data["phases"]=phasevector
            # convert the geometries into bohr
            #atoms=ase.atoms.Atoms(geoms.get_atomic_numbers(),geoms.get_positions()/Bohr)
        if "forces" not in data:
            data["has_forces"]=np.array[0].reshape(-1)
            data["forces"]= np.zeros((nsinglets+ntriplets,natoms,3))

        return data

def interpolate(start, end,natoms,n_singlets,n_triplets,n_int):
    '''This file should interpolate between a starting geometry taken from a QM.in file and an end-geometry (at which NNs broke down), also taken from a QM.in file_end
          - The files should be transfered into zmat-files and linear interpolation between the start- and end-geometry will be carried out
          - Later, the geometries will be written into the form of a QM.in file, the interface will then generate QM.out files - those will be written into an output.dat format and the output_all.dat can be appended later
          - In the end, the phases should be compared between geometries and corrected in the output.dat files. After the last geometry is corrected, the 
            calculation using a QM-interface, should be carried out with corrected phases
    '''


    #============================================================================================================================================================#
    #================================================================INTERPOLATION OF ZMAT-Files=================================================================#
    #============================================================================================================================================================#
  #first change xyz files to Zmat files
    os.system("obabel -ixyz %s -ogzmat -O start.zmat" %(start))
    os.system("obabel -ixyz %s -ogzmat -O end.zmat" %(end))
    os.system("tail -n +6 start.zmat > start_1.zmat" ) 
    os.system("tail -n +6 end.zmat > end_1.zmat" )
    #nmstates = Restart_Properties['nmstates']
    #natoms = Restart_Properties['natoms']
    natoms = natoms
    nmstates = 33
    #read files
    scanpath=os.getcwd()
    f1=open(scanpath+'/start_1.zmat', 'r')
    data1=f1.readlines()
    f1.close()

    f2=open(scanpath+'/end_1.zmat', 'r')
    data2=f2.readlines()
    f2.close()

    #parse files
    res1=read_zmat(data1)
    res2=read_zmat(data2)
    #if res1['header']!=res2['header']:
      #print 'Problem with headers!'
      #sys.exit(1)

    #prepare interpolation
    base = 'interpolate'
    nsteps=int(n_int)
    #iinterpolate and write files
    for istep in range(nsteps):
      string='#\n\ntitle\n\n0 0\n'+''.join(res1['header'])
      string+='\n'
      for i in res1:
        if i=='header':
          continue
        f=interpol(res1[i],res2[i],istep,nsteps)
        string+='%s %20.12f\n' % (i,f)
      string+='\n\n\n'
      filename=scanpath+'/'+base+str(istep)+'.zmat'
      fi=open(filename,'w')
      fi.write(string)
      fi.close

    #changes files from zmat to xyz-files and append files to make valid QM.in
    i=int(0)
    laststep=int(nsteps)-1
    while True:
      if i<laststep:
        os.system("obabel -igzmat %s/interpolate%i.zmat -oxyz -O %s/interpolate%i.xyz" %(scanpath,i,scanpath,i))
        #make alignment to first xyz-file
        if i==0:
          #make an alignment-file once
          write_alignsh(natoms,scanpath)
          #read QM.out from starting geometry (ICOND_00000) to get the initial velocity vector
          #Properties['0']={}
          #Properties=read_QMinit_out(scanpath,Properties)
        cwd=os.getcwd()
        os.system("bash %s/align.sh %s/start.xyz %s/%s" %(scanpath, scanpath, scanpath, base+str(i)+'.xyz'))
        #append files with QM.in string
        file=open(scanpath+'/interpolate'+str(i)+'.xyz', 'a')
        #file.write(string_QMin)
        file.close()
        i+=1
      elif i>=laststep:
        break
    #make the last file separately since it is not working otherwise
    os.system("obabel -igzmat %s/end.zmat -oxyz -O %s/interpolate%i.xyz" %(scanpath,scanpath,laststep))
    os.system("bash %s/align.sh %s/start.xyz %s/%s" %(scanpath,scanpath,scanpath,base+str(laststep)+'.xyz'))
    #file = open(scanpath+'/interpolate'+str(laststep)+'.xyz','a')
    #file.write(string_QMin)
    #file.close()
    #qm_calc(scanpath, laststep,n_singlets,n_triplets)
    #Properties,OUTPUT=phasecorrection(Properties, QMpath, NNpath, scanpath, laststep, NNpath_superior, nsteps)
    #generate one output.dat file containing all interpolated geometries
    #generate_outputdat(scanpath+'/QM/', nsteps)
    #generate_phasevector(Properties,laststep,trajpath,nmstates)
    #rm QM.out files
    #TODO put # away of the 2 following lines
    #for i in range(1,nsteps):
      #os.system( 'rm %s/QM/QM%i~.out %s/QM/QM%i.out' %(scanpath,i,scanpath,i))

   
def read_zmat(data):
    res={'header':[]}
    for iline,line in enumerate(data):
      if line.strip()=='':
        res['header'].append(line)
      else:
        break
    for iline, line in enumerate(data):
    #while True:
      if iline==len(data):
        break
      line=data[iline]
      if line.startswith('Variables:'):
        iline+=1
        for index in range(iline,len(data)-1):
          line=data[index]
          s=line.split()
          res[s[0]]=float(s[1])
        break
      else:
        res['header'].append(line)
      iline+=1
    return res

def interpol(x,y,istep,nsteps):
    short_angle=((y-x)+180.)%360.-180.
    #print short_angle
    return x + short_angle*istep/(nsteps-1)

def write_alignsh(natoms,scanpath):
    string=''
    natoms_2=int(natoms)-int(1)
    string+='n=%i\n' %natoms
    string+='m=$(echo "$n-1"|bc)\n'
    string+='q="index 0 to %i"\n' %natoms_2
    string+='xyz1=$1\n'
    string+='xyz2=$2\n\n'
    string+='#write VMD input\n'
    string+='echo "\n'
    string+='mol new \%s$2\%s\n' %('"','"')
    string+='mol new \%s$1\%s\n\n'%('"','"')
    string+='set sel0 [atomselect 0 \%s$q\%s]\n' %('"','"')
    string+='set sel0_ [atomselect 0 \%sall\%s]\n'%('"','"')
    string+='set sel1 [atomselect 1 \%s$q\%s]\n'%('"','"')
    string+='set M [measure fit \$sel0 \$sel1]\n'
    string+='\$sel0_ move \$M\n\n'
    string+='\$sel0_ writepdb \%stmp.pdb\%s\n'%('"','"')
    string+='quit\n'
    string+='" > tmp.vmd\n\n'
    string+='#run VMD\n'
    string+='vmd -e tmp.vmd -dispdev text $> /dev/null\n'
    string+='rm tmp.vmd\n\n'
    string+='#convert with obabel\n'
    string+='obabel -ipdb tmp.pdb -oxyz -O $2\n'
    string+='rm tmp.pdb\n'
    alignfile=open(scanpath+'/align.sh', 'w')
    alignfile.write(string)
    alignfile.close()
def correct_phases(data,nsinglets,ntriplets):
    nstates = nsinglets+3*ntriplets
    corrected_data = {}
    phases=np.ones((nsinglets+ntriplets))
    dataiterator=-1
    for i in range(len(data)):
        
        # skip correction if no phase vector was found
        # only if we don't interpolate
        
        if "phases" not in data[i]:
            pass
        else:
            dataiterator+=1 
            corrected_data[dataiterator]={}
            #Energies and forces do not need to be corrected
            if "energy" in data[i]:
                corrected_data[dataiterator]["energy"] = data[i]["energy"]
                
            if "forces" in data[i]:
                corrected_data[dataiterator]["forces"] = data[i]["forces"]
                corrected_data[dataiterator]["has_forces"] = data[i]["has_forces"]
            
            #socs
            if "socs" in data[i]:
                corrected_data[dataiterator]["socs"] = data[i]["socs"]
                sociterator=-1
                
                for istate in range(nstates*2):

                    for jstate in range(2+istate*2,nstates*2):
                        if istate < (nsinglets+ntriplets): 
                            state1 = istate
                        if jstate < (nsinglets+ntriplets):
                            state2 = jstate
                        if istate >= (nsinglets+ntriplets) and istate < (nsinglets+2*ntriplets):
                            state1 = istate - ntriplets
                        if istate >= (nsinglets+ntriplets*2):
                            state1 = istate - 2*ntriplets
                        if jstate >= (nsinglets+ntriplets) and istate < (nsinglets+2*ntriplets):
                            state2 = jstate - ntriplets
                        if jstate >= (nsinglets+2*ntriplets):
                            state2 = jstate - ntriplets*2
                        sociterator+=1
                        corrected[data]["socs"][sociterator] = data[i]["socs"] * phases[state1] * phases[state2]
            # Nonadiabatic couplings
            
            if "nacs" in data[i]:
                naciterator=-1
                corrected_data[dataiterator]["nacs"] = data[i]["nacs"]
                for inac1 in range(nsinglets):
                    for inac2 in range(inac1+1,nsinglets):
                        naciterator+=1
                        corrected_data[dataiterator]["nacs"][naciterator]=data[i]["nacs"][naciterator] * phases[inac1] * phases[inac2]
                for inac1 in range(nsinglets,nsinglets+ntriplets):
                    for inac2 in range(inac1+1,nsinglets+ntriplets):
                        naciterator+=1
                        corrected_data[dataiterator]["nacs"][naciterator]=data[i]["nacs"][naciterator] * phases[inac1] * phases[inac2]
             
            # dipoles
            # only offdiagonal elements are corrected
            
            if "dipoles" in data[i]:
                dipoleiterator=-1
                corrected_data[dataiterator]["dipoles"] = data[i]["dipoles"]
                for idipole1 in range(nsinglets):
                    for idipole2 in range(nsinglets):
                        if idipole2< idipole1:
                            pass
                        else:
                            dipoleiterator+=1
                            corrected_data[dataiterator]["dipoles"][dipoleiterator]=data[i]["dipoles"][dipoleiterator] * phases[idipole1] * phases[idipole2]
                for idipole1 in range(nsinglets,nsinglets+ntriplets):
                    for idipole2 in range(nsinglets,nsinglets+ntriplets):
                        if idipole2< idipole1:
                            pass
                        else:
                            dipoleiterator+=1
                            corrected_data[dataiterator]["dipoles"][dipoleiterator]=data[i]["dipoles"][dipoleiterator] * phases[idipole1] * phases[idipole2]
                                     
                            
    return corrected_data




def read_traj(path,nsinglets,ntriplets,threshold):
    try:
        trajfile = open(path+"/output.dat","r").readlines()
        geomfile = open(path+"/output.xyz","r").readlines()
        if len(geomfile) <=1 or len(trajfile) <=1:
            print("No file output.dat or output.xyz given in the specified folder. Please make sure you have converted all data with SHARC.")
            
    except IOError:
        print("No file output.dat or output.xyz given in the specified folder. Please make sure you have converted all data with SHARC.")
         
    lineiterator=-1
    trajfile = open("%s/output.dat"%path,"r").readlines()
    for line in trajfile:
        lineiterator+=1
        if "nsteps" in line:
            nsteps = int(line.split()[1])
        if "natom" in line:
            natoms = int(line.split()[1])
            
        if "ezero" in line:
            ezero = float(line.split()[1])
            
        """if line.startswith("! Atomic Numbers"):
            atypes = []
            
            for iatom in range(len(natoms)):
                atypes.append(int(trajfile[lineiterator+1+iatom].split()[0]))
            print(atypes)"""
        if lineiterator>=100:
            break
    data={}
    atoms=ase.io.read("%s/output.xyz"%path,":")

    #number of nacs
    nnacs = int(nsinglets*(nsinglets-1)/2) + int(ntriplets*(ntriplets-1)/2)
    #number of dipole moment values
    ndipoles = int(nsinglets+ntriplets+nnacs)
    #nstates
    nstates = nsinglets+3*ntriplets
    iterator=-1
    skip_overlap = False
    istep = -1 
    
    for line in trajfile:
        iterator+=1
        
        if line.startswith("! 0 Step"):
            
            has_forces = False
            has_nacs = False
            istep+=1
            found_overlap=False
            #each data point gets one entry in the dictionary
            data[istep] = {}
            data[istep]["has_forces"]=np.array([1]).reshape(-1)
     
        if line.startswith("! 1 Hamiltonian (MCH) in a.u."):
            energies = np.zeros((nsinglets+ntriplets,1))
            for istate in range(nsinglets+ntriplets):
                energyline = trajfile[iterator+1+istate].split()
                energies[istate]=float(energyline[2*istate])
            data[istep]["energy"]=energies.reshape(-1)
            

            #get spin-orbit couplings
            if ntriplets != 0:
                
                #triangular minus diagonal (times two due to complex values)
                socs = np.zeros((int(nstates*nstates-nstates)))
                sociterator = -1
                for istate in range(nstates*2):
                    socsline = trajfile[iterator+1+istate].split()
                    for jstate in range(2+istate*2,2*nstates):
                        sociterator+=1
                        socs[sociterator]=float(socsline[jstate])
                data[istep]["socs"] = socs

            
        if line.startswith("! 15 Gradients (MCH)") and has_forces == False:
            has_forces = True
            forces = np.zeros((nsinglets+ntriplets,natoms,3))
            for istate in range(nsinglets+ntriplets):
                for iatom in range(natoms):
                    forceline = trajfile[iterator+1+istate+istate*natoms+iatom].split()
                    for xyz in range(3):
                        forces[istate][iatom][xyz]=float(forceline[xyz])
            data[istep]["forces"]=forces

        if line.startswith("! 16 NACdr matrix element (MCH)") and has_nacs == False:
            has_nacs = True
            
         
            # Getting nonadiabatic couplings (NACs)
            # Only take the ones that are between different states, so in this case: skip NAC(1,1)=0
            # Note that only one NAC is saved as NAC(1,2)=-NAC(2,1) - numbers in brackets denote states
            nacs = np.zeros((nnacs,natoms,3))
            naciterator=-1
            for inac1 in range(nstates):
                for inac2 in range(nstates):
                    if inac1==inac2  or inac2<inac1 or (inac1 < nsinglets and inac2>=nsinglets) or inac2>=(nsinglets+ntriplets) or inac1 >=(nsinglets+ntriplets):
                        pass
                    else:
                        naciterator+=1
                        for iatom in range(natoms):
                            nacline = trajfile[iterator+1+inac1*nstates+inac1*natoms*nstates+inac2*natoms+inac2+iatom].split()
                            for xyz in range(3):
                                nacs[naciterator][iatom][xyz] = float(nacline[xyz])
            data[istep]["nacs"]=nacs
   

        if line.startswith("! 3 Dipole moments X (MCH) in a.u."):
            
            # Getting values for dipole moments
            # Transition and permanent dipole moments are included in x,y,z direction (last dimension)
            # The values are read line after line starting from the diagonal values
            # In the QM.out file, there are 3 blocks, one block belongs to one direction (x,y,z)
            # SchNarc has ( ndipoles x (xyz) ) as a shape
            dipoles = np.zeros((ndipoles,3))
            for xyz in range(3):
                dipoleiterator=-1
                for idipole1 in range(nstates):
                    for idipole2 in range(nstates):                          
                        if idipole2<idipole1 or (idipole1 < (nsinglets) and idipole2>=nsinglets) or idipole2>=(nsinglets+ntriplets) or idipole1 >=(nsinglets+ntriplets):
                            pass
                        else:
                            dipoleiterator+=1
                            dipoleline = trajfile[iterator+1+xyz+xyz*nstates+idipole1].split()
                            dipoles[dipoleiterator][xyz] = float(dipoleline[idipole2*2])
            data[istep]["dipoles"]=dipoles
            
        if line.startswith("! 4 Overlap matrix (MCH)") and skip_overlap == False:
            phasevector = np.ones((nsinglets+ntriplets))
            found_overlap = False

            overlapiterator=-1
            for istate in range(nsinglets+ntriplets):
                overlapiterator+=1
                overlapline =trajfile[iterator+1+overlapiterator].split()
                #skip imaginary values (all 0)
                #this asks about the diagonal values of the overlap matrix
                if np.abs(float(overlapline[2*istate])) >= threshold:
                    found_overlap=True
                    if float(overlapline[2*istate]) >= threshold:
                        phasevector[istate] = +1
                    else:
                        phasevector[istate] = -1
                else:
                    # if the overlap are not large enough for one state, then it could be that states have switched.
                    # we will check for this below
                    # this asks about all entries
                    for jstate in range(nsinglets+ntriplets):
                        if np.abs(float(overlapline[2*jstate])) >= threshold:
                            found_overlap = True
                            if float(overlapline[2*jstate]) >= threshold:
                                phasevector[istate] = +1
                            else:
                                phasevector[istate] = -1
            if found_overlap == True:
                data[istep]["phases"]=phasevector
            if found_overlap == False:
                skip_overlap = True
                print("Overlaps could only be found until step ", istep, "but data points are saved nevertheless")
                
        
    return data, atoms
