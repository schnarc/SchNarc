#!/usr/bin/env python2

# Interactive script for the setup of dynamics calculations for SHARC
# 
# usage: python setup_traj.py

import copy
import math
import sys
import re
import os
import stat
import shutil
import datetime
import random
from optparse import OptionParser
import readline
import time
from socket import gethostname
import ast
from generate_QMin import gen_QMin

# =========================================================0
# compatibility stuff

if sys.version_info[0]!=2:
  print 'This is a script for Python 2!'
  sys.exit(0)

if sys.version_info[1]<5:
  def any(iterable):
    for element in iterable:
      if element:
        return True
    return False

  def all(iterable):
    for element in iterable:
      if not element:
        return False
    return True


# some constants
DEBUG = False
CM_TO_HARTREE = 1./219474.6     #4.556335252e-6 # conversion factor from cm-1 to Hartree
HARTREE_TO_EV = 27.211396132    # conversion factor from Hartree to eV
U_TO_AMU = 1./5.4857990943e-4            # conversion from g/mol to amu
BOHR_TO_ANG=0.529177211
PI = math.pi

version='2.0'
versionneeded=[0.2, 1.0, 2.0]
versiondate=datetime.date(2017,3,1)


IToMult={
         1: 'Singlet', 
         2: 'Doublet', 
         3: 'Triplet', 
         4: 'Quartet', 
         5: 'Quintet', 
         6: 'Sextet', 
         7: 'Septet', 
         8: 'Octet', 
         'Singlet': 1, 
         'Doublet': 2, 
         'Triplet': 3, 
         'Quartet': 4, 
         'Quintet': 5, 
         'Sextet': 6, 
         'Septet': 7, 
         'Octet': 8
         }


Couplings={
  1: {'name':        'nacdt',
      'description': 'DDT     =  < a|d/dt|b >        Hammes-Schiffer-Tully scheme   '
     },
  2: {'name':        'nacdr',
      'description': 'DDR     =  < a|d/dR|b >        Original Tully scheme          '
     },
  3: {'name':        'overlap',
      'description': 'overlap = < a(t0)|b(t) >       Local Diabatization scheme     '
     }
  }

EkinCorrect={
  1: {'name':             'none',
      'description':      'Do not conserve total energy. Hops are never frustrated.',
      'description_refl': 'Do not reflect at a frustrated hop.',
      'required':   []
     },
  2: {'name':             'parallel_vel',
      'description':      'Adjust kinetic energy by rescaling the velocity vectors. Often sufficient.',
      'description_refl': 'Reflect the full velocity vector.',
      'required':   []
     },
  3: {'name':             'parallel_nac',
      'description':      'Adjust kinetic energy only with the component of the velocity vector along the non-adiabatic coupling vector.',
      'description_refl': 'Reflect only the component of the velocity vector along the non-adiabatic coupling vector.',
      'required':   ['nacdr']
     }
  }

Decoherences={
  1: {'name':             'none',
      'description':      'No decoherence correction.',
      'required':   [],
      'params':     ''
     },
  2: {'name':             'edc',
      'description':      'Energy-based decoherence scheme (Granucci, Persico, Zoccante).',
      'required':   [],
      'params':     '0.1'
     },
  3: {'name':             'afssh',
      'description':      'Augmented fewest-switching surface hopping (Jain, Alguire, Subotnik).',
      'required':   [],
      'params':     ''
     }
  }

HoppingSchemes={
  1: {'name':             'off',
      'description':      'Surface hops off.'
     },
  2: {'name':             'sharc',
      'description':      'Standard SHARC surface hopping probabilities (Mai, Marquetand, Gonzalez).'
     },
  3: {'name':             'gfsh',
      'description':      'Global flux surface hopping probabilities (Wang, Trivedi, Prezhdo).'
     }
  }

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def try_read(l,index,typefunc,default):
  try:
    if typefunc==bool:
      return 'True'==l[index]
    else:
      return typefunc(l[index])
  except IndexError:
    return typefunc(default)
  except ValueError:
    print 'Could not initialize object!'
    quit(1)

# ======================================================================================================================

class ATOM:
  def __init__(self,symb='??',num=0.,coord=[0.,0.,0.],m=0.,veloc=[0.,0.,0.]):
    self.symb  = symb
    self.num   = num
    self.coord = coord
    self.mass  = m
    self.veloc = veloc
    self.Ekin=0.5*self.mass * sum( [ self.veloc[i]**2 for i in range(3) ] )

  def init_from_str(self,initstring=''):
    f=initstring.split()
    self.symb  =   try_read(f,0,str,  '??')
    self.num   =   try_read(f,1,float,0.)
    self.coord = [ try_read(f,i,float,0.) for i in range(2,5) ]
    self.mass  =   try_read(f,5,float,0.)*U_TO_AMU
    self.veloc = [ try_read(f,i,float,0.) for i in range(6,9) ]
    self.Ekin=0.5*self.mass * sum( [ self.veloc[i]**2 for i in range(3) ] )

  def __str__(self):
    s ='%2s % 5.1f '               % (self.symb, self.num)
    s+='% 12.8f % 12.8f % 12.8f '  % tuple(self.coord)
    s+='% 12.8f '                  % (self.mass/U_TO_AMU)
    s+='% 12.8f % 12.8f % 12.8f'   % tuple(self.veloc)
    return s

  def EKIN(self):
    self.Ekin=0.5*self.mass * sum( [ self.veloc[i]**2 for i in range(3) ] )
    return self.Ekin

  def geomstring(self):
    s='  %2s % 5.1f % 12.8f % 12.8f % 12.8f % 12.8f' % (self.symb,self.num,self.coord[0],self.coord[1],self.coord[2],self.mass/U_TO_AMU)
    return s

  def velocstring(self):
    s=' '*11+'% 12.8f % 12.8f % 12.8f' % tuple(self.veloc)
    return s

# ======================================================================================================================

class STATE:
  def __init__(self,i=0,e=0.,eref=0.,dip=[0.,0.,0.]):
    self.i       = i
    self.e       = e.real
    self.eref    = eref.real
    self.dip     = dip
    self.Excited = False
    self.Eexc    = self.e-self.eref
    self.Fosc    = (2./3.*self.Eexc*sum( [i*i.conjugate() for i in self.dip] ) ).real
    if self.Eexc==0.:
      self.Prob  = 0.
    else:
      self.Prob  = self.Fosc/self.Eexc**2

  def init_from_str(self,initstring):
    f=initstring.split()
    self.i       =   try_read(f,0,int,  0 )
    self.e       =   try_read(f,1,float,0.)
    self.eref    =   try_read(f,2,float,0.)
    self.dip     = [ complex( try_read(f,i,float,0.),try_read(f,i+1,float,0.) ) for i in [3,5,7] ]
    self.Excited =   try_read(f,11,bool, False)
    self.Eexc    = self.e-self.eref
    self.Fosc    = (2./3.*self.Eexc*sum( [i*i.conjugate() for i in self.dip] ) ).real
    if self.Eexc==0.:
      self.Prob  = 0.
    else:
      self.Prob  = self.Fosc/self.Eexc**2

  def __str__(self):
    s ='%03i % 18.10f % 18.10f ' % (self.i,self.e,self.eref)
    for i in range(3):
      s+='% 12.8f % 12.8f ' % (self.dip[i].real,self.dip[i].imag)
    s+='% 12.8f % 12.8f %s' % (self.Eexc*HARTREE_TO_EV,self.Fosc,self.Excited)
    return s

  def Excite(self,max_Prob,erange):
    try:
      Prob=self.Prob/max_Prob
    except ZeroDivisionError:
      Prob=-1.
    if not (erange[0] <= self.Eexc <= erange[1]):
      Prob=-1.
    self.Excited=(random.random() < Prob)

# ======================================================================================================================

class INITCOND:
  def __init__(self,atomlist=[],eref=0.,epot_harm=0.):
    self.atomlist=atomlist
    self.eref=eref
    self.Epot_harm=epot_harm
    self.natom=len(atomlist)
    self.Ekin=sum( [atom.Ekin for atom in self.atomlist] )
    self.statelist=[]
    self.nstate=0
    self.Epot=epot_harm

  def addstates(self,statelist):
    self.statelist=statelist
    self.nstate=len(statelist)
    self.Epot=self.statelist[0].e-self.eref

  def init_from_file(self,f,eref,index):
    while True: 
      line=f.readline()
      #if 'Index     %i' % (index) in line:
      if re.search('Index\s+%i' % (index),line):
        break
      if line=='\n':
        continue
      if line=='':
        print 'Initial condition %i not found in file %s' % (index,f.name)
        quit(1)
    f.readline()        # skip one line, where "Atoms" stands
    atomlist=[]
    while True:
      line=f.readline()
      if 'States' in line:
        break
      atom=ATOM()
      atom.init_from_str(line)
      atomlist.append(atom)
    statelist=[]
    while True:
      line=f.readline()
      if 'Ekin' in line:
        break
      state=STATE()
      state.init_from_str(line)
      statelist.append(state)
    epot_harm=0.
    while not line=='\n' and not line=='':
      line=f.readline()
      if 'epot_harm' in line.lower():
        epot_harm=float(line.split()[1])
        break
    self.atomlist=atomlist
    self.eref=eref
    self.Epot_harm=epot_harm
    self.natom=len(atomlist)
    self.Ekin=sum( [atom.Ekin for atom in self.atomlist] )
    self.statelist=statelist
    self.nstate=len(statelist)
    if self.nstate>0:
      self.Epot=self.statelist[0].e-self.eref
    else:
      self.Epot=epot_harm

  def __str__(self):
    s='Atoms\n'
    for atom in self.atomlist:
      s+=str(atom)+'\n'
    s+='States\n'
    for state in self.statelist:
      s+=str(state)+'\n'
    s+='Ekin      % 16.12f a.u.\n' % (self.Ekin)
    s+='Epot_harm % 16.12f a.u.\n' % (self.Epot_harm)
    s+='Epot      % 16.12f a.u.\n' % (self.Epot)
    s+='Etot_harm % 16.12f a.u.\n' % (self.Epot_harm+self.Ekin)
    s+='Etot      % 16.12f a.u.\n' % (self.Epot+self.Ekin)
    s+='\n\n'
    return s



# ======================================================================================================================

def centerstring(string,n,pad=' '):
  l=len(string)
  if l>=n:
    return string
  else:
    return  pad*((n-l+1)/2)+string+pad*((n-l)/2)

def displaywelcome():
  print 'Script for setup of initial conditions started...\n'
  string='\n'
  string+='  '+'='*80+'\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='||'+centerstring('Setup trajectories for SHARC dynamics',80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='||'+centerstring('Author: Sebastian Mai, Philipp Marquetand',80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='||'+centerstring('Version:'+version,80)+'||\n'
  string+='||'+centerstring(versiondate.strftime("%d.%m.%y"),80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='  '+'='*80+'\n\n'
  string+='''
This script automatizes the setup of the input files for SHARC dynamics. 
  '''
  print string

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def open_keystrokes():
  global KEYSTROKES
  KEYSTROKES=open('KEYSTROKES.tmp','w')

def close_keystrokes():
  KEYSTROKES.close()
  shutil.move('KEYSTROKES.tmp','KEYSTROKES.setup_traj')

# ===================================

def question(question,typefunc,default=None,autocomplete=True,ranges=False):
  if typefunc==int or typefunc==float:
    if not default==None and not isinstance(default,list):
      print 'Default to int or float question must be list!'
      quit(1)
  if typefunc==str and autocomplete:
    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind("tab: complete")    # activate autocomplete
  else:
    readline.parse_and_bind("tab: ")            # deactivate autocomplete

  while True:
    s=question
    if default!=None:
      if typefunc==bool or typefunc==str:
        s+= ' [%s]' % (str(default))
      elif typefunc==int or typefunc==float:
        s+= ' ['
        for i in default:
          s+=str(i)+' '
        s=s[:-1]+']'
    if typefunc==str and autocomplete:
      s+=' (autocomplete enabled)'
    if typefunc==int and ranges:
      s+=' (range comprehension enabled)'
    s+=' '

    line=raw_input(s)
    line=re.sub('#.*$','',line).strip()
    if not typefunc==str:
      line=line.lower()

    if line=='' or line=='\n':
      if default!=None:
        KEYSTROKES.write(line+' '*(40-len(line))+' #'+s+'\n')
        return default
      else:
        continue

    if typefunc==bool:
      posresponse=['y','yes','true', 't', 'ja',  'si','yea','yeah','aye','sure','definitely']
      negresponse=['n','no', 'false', 'f', 'nein', 'nope']
      if line in posresponse:
        KEYSTROKES.write(line+' '*(40-len(line))+' #'+s+'\n')
        return True
      elif line in negresponse:
        KEYSTROKES.write(line+' '*(40-len(line))+' #'+s+'\n')
        return False
      else:
        print 'I didn''t understand you.'
        continue

    if typefunc==str:
      KEYSTROKES.write(line+' '*(40-len(line))+' #'+s+'\n')
      return line

    if typefunc==float:
      # float will be returned as a list
      f=line.split()
      try:
        for i in range(len(f)):
          f[i]=typefunc(f[i])
        KEYSTROKES.write(line+' '*(40-len(line))+' #'+s+'\n')
        return f
      except ValueError:
        print 'Please enter floats!'
        continue

    if typefunc==int:
      # int will be returned as a list
      f=line.split()
      out=[]
      try:
        for i in f:
          if ranges and '~' in i:
            q=i.split('~')
            for j in range(int(q[0]),int(q[1])+1):
              out.append(j)
          else:
            out.append(int(i))
        KEYSTROKES.write(line+' '*(40-len(line))+' #'+s+'\n')
        return out
      except ValueError:
        if ranges:
          print 'Please enter integers or ranges of integers (e.g. "-3~-1  2  5~7")!'
        else:
          print 'Please enter integers!'
        continue

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def itnmstates(states):
  for i in range(len(states)):
    if states[i]<1:
      continue
    for k in range(i+1):
      for j in range(states[i]):
        yield i+1,j+1,k-i/2.
  return

# ======================================================================================================================


class init_string:
  def __init__(self):
    self.strings=[]
    self.nst=0
    self.width=100
    self.group=10
    self.groups=(self.width-1)/self.group+1
    self.nrow=1
    self.lastrow=0
  def add(self,s):
    self.strings.append(s)
    self.nst+=1
    self.nrow=(self.nst-1)/self.width+1
    self.lastrow=self.nst%self.width
    if self.lastrow==0:
      self.lastrow=self.width
  def reset(self):
    self.strings=[]
    self.nst=0
    self.nrow=1
    self.lastrow=0
  def __str__(self):
    nw=int(math.log(self.nst)/math.log(10)+1.1)
    s=' '*(nw+2)
    fs='%%%ii' % (nw)
    for i in range(self.groups):
      s+=' '*(self.group-nw+1)+fs % ((i+1)*self.group)
    s+='\n'
    s+=' '*(nw+2)
    for i in range(self.groups):
      s+=' '
      for j in range(self.group-1):
        s+=' '
      s+='|'
    s+='\n'
    index=0
    for i in range(self.nrow):
      s+=fs % (i*self.width) + ' | '
      for j in range(self.width):
        try:
          s+=self.strings[index]
        except IndexError:
          return s
        index+=1
        if (j+1)%self.group==0:
          s+=' '
      s+='\n'
    s+='\n'
    return s

# ======================================================================================================================

def analyze_initconds(initlist,INFOS):
  if INFOS['show_content']:
    print 'Contents of the initconds file:'
    print '''\nLegend:
?       Geometry and Velocity
.       not selected
#       selected
'''
  n_hasexc=[]
  n_issel=[]
  display=init_string()
  for state in range(INFOS['nstates']):
    if INFOS['show_content']:
      print 'State %i:' % (state+1)
    display.reset()
    n_hasexc.append(0)
    n_issel.append(0)
    for i in initlist:
      if len(i.statelist)<state+1:
        display.add('?')
      else:
        n_hasexc[-1]+=1
        if i.statelist[state].Excited:
          display.add('#')
          n_issel[-1]+=1
        else:
          display.add('.')
    if INFOS['show_content']:
      print display
  print 'Number of excited states and selections:'
  print   'State    #InitCalc       #Selected'
  for i in range(len(n_hasexc)):
    s= '% 5i        % 5i           % 5i' % (i+1,n_hasexc[i],n_issel[i])
    if not INFOS['isactive'][i]:
      s+='  inactive'
    print s
  return n_issel

# ======================================================================================================================

def get_initconds(INFOS):
  ''''''

  INFOS['initf'].seek(0)                 # rewind the initf file
  initlist=[]
  for icond in range(1,INFOS['ninit']+1):
    initcond=INITCOND()
    initcond.init_from_file(INFOS['initf'],INFOS['eref'],icond)
    initlist.append(initcond)
  print 'Number of initial conditions in file:       %5i' % (INFOS['ninit'])

  INFOS['initlist']=initlist
  INFOS['n_issel']=analyze_initconds(initlist,INFOS)
  return INFOS


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def get_general():
  '''This routine questions from the user some general information:
  - initconds file
  - number of states
  - number of initial conditions
  - interface to use'''

  INFOS={}
  string='\n  '+'='*80+'\n'
  string+='||'+centerstring('Initial conditions',80)+'||\n'
  string+='  '+'='*80+'\n\n'
  print string
  print '''\nThis script reads the initial conditions (geometries, velocities, initial excited state)
from the initconds.excited files as provided by excite.py. 
'''

  # open the initconds file
  try:
    initfile='initconds.excited'
    initf=open(initfile)
    INFOS['initcondsexcited']="%s/initconds.excited" %(os.getcwd())
    line=initf.readline()
  except IOError:
    print 'Please enter the filename of the initial conditions file.'
    while True:
      initfile=question('Initial conditions filename:',str,'initconds.excited')
      initfile=os.path.expanduser(os.path.expandvars(initfile))
      if os.path.isdir(initfile):
        print 'Is a directory: %s' % (initfile)
        continue
      if not os.path.isfile(initfile):
        print 'File does not exist: %s' % (initfile)
        continue
      try:
        initf=open(initfile,'r')
      except IOError:
        print 'Could not open: %s' % (initfile)
        continue
      line=initf.readline()
  # read the header
  INFOS['ninit']=int(initf.readline().split()[1])
  INFOS['natom']=int(initf.readline().split()[1])
  print '\nFile %s contains %i initial conditions.' % (initfile,INFOS['ninit'])
  print 'Number of atoms is %i' % (INFOS['natom'])
  INFOS['repr']=initf.readline().split()[1]
  if INFOS['repr']=='MCH':
    INFOS['diag']=False
  else:
    INFOS['diag']=True
  INFOS['eref']=float(initf.readline().split()[1])
  INFOS['eharm']=float(initf.readline().split()[1])

  # get guess for number of states
  line=initf.readline()
  if 'states' in line.lower():
    states=[]
    l=line.split()
    for i in range(1,len(l)):
      states.append(int(l[i]))
    guessstates=states
  else:
    guessstates=None

  print 'Reference energy %16.12f a.u.' % (INFOS['eref'])
  print 'Excited states are in %s representation.\n' % (['MCH','diagonal'][INFOS['diag']])
  initf.seek(0)                 # rewind the initf file
  INFOS['initf']=initf


  # Number of states
  print '\nPlease enter the number of states as a list of integers\ne.g. 3 0 3 for three singlets, zero doublets and three triplets.'
  while True:
    states=question('Number of states:',int,guessstates)
    if len(states)==0:
      continue
    if any(i<0 for i in states):
      print 'Number of states must be positive!'
      continue
    break
  print ''
  nstates=0
  for mult,i in enumerate(states):
    nstates+=(mult+1)*i
  print 'Number of states: '+str(states)
  print 'Total number of states: %i\n' % (nstates)
  INFOS['states']=states
  INFOS['nstates']=nstates
  # obtain the statemap 
  statemap={}
  i=1
  for imult,istate,ims in itnmstates(INFOS['states']):
    statemap[i]=[imult,istate,ims]
    i+=1
  INFOS['statemap']=statemap

  # get active states
  if question('Do you want all states to be active?',bool,True):
    INFOS['actstates']=INFOS['states']
  else:
    print '\nPlease enter the number of ACTIVE states as a list of integers\ne.g. 3 0 3 for three singlets, zero doublets and three triplets.'
    while True:
      actstates=question('Number of states:',int)
      if len(actstates)!=len(INFOS['states']):
        print 'Length of nstates and actstates must match!'
        continue
      valid=True
      for i,nst in enumerate(actstates):
        if not 0<=nst<=INFOS['states'][i]:
          print 'Number of active states of multiplicity %i must be between 0 and the number of states of this multiplicity (%i)!' % (i+1,INFOS['states'][i])
          valid=False
      if not valid:
        continue
      break
    INFOS['actstates']=actstates
  isactive=[]
  for imult in range(len(INFOS['states'])):
    for ims in range(imult+1):
      for istate in range(INFOS['states'][imult]):
        isactive.append( (istate+1<=INFOS['actstates'][imult]) )
  INFOS['isactive']=isactive
  print ''


  # ask whether initfile content is shown
  INFOS['show_content']=question('Do you want to see the content of the initconds file?',bool,True)



  # read initlist, analyze it and print content (all in get_initconds)
  INFOS['initf']=initf
  INFOS=get_initconds(INFOS)


  # Generate random example for setup-states, according to Leti's wishes
  exampleset=set()
  nactive=sum(INFOS['isactive'])
  while len(exampleset)<min(3,nactive):
    i=random.randint(1,INFOS['nstates'])
    if INFOS['isactive'][i-1]:
      exampleset.add(i)
  exampleset=list(exampleset)
  exampleset.sort()
  string1=''
  string2=''
  j=0
  for i in exampleset:
    j+=1
    if j==len(exampleset) and len(exampleset)>1:
      string1+=str(i)
      string2+='and '+str(i)
    else:
      string1+=str(i)+' '
      string2+=str(i)+', '



  # ask for states to setup
  print '\nPlease enter a list specifying for which excited states trajectories should be set-up\ne.g. %s to select states %s.' % (string1,string2)
  defsetupstates=[]
  nmax=0
  for i,active in enumerate(INFOS['isactive']):
    if active and INFOS['n_issel'][i]>0:
      defsetupstates.append(i+1)
      nmax+=INFOS['n_issel'][i]
  if nmax<=0:
    print '\nZero trajectories can be set up!'
    sys.exit(1)
  while True:
    setupstates=question('States to setup the dynamics:',int,defsetupstates,ranges=True)
    valid=True
    for i in setupstates:
      if i>INFOS['nstates']:
        print 'There are only %i states!' % (INFOS['nstates'])
        valid=False
        continue
      if i<0:
        valid=False
        continue
      if not INFOS['isactive'][i-1]:
        print 'State %i is inactive!' % (i)
        valid=False
    if not valid:
      continue
    INFOS['setupstates']=set(setupstates)
    nsetupable=sum( [ INFOS['n_issel'][i-1] for i in INFOS['setupstates'] if INFOS['isactive'][i-1] ] )
    print '\nThere can be %i trajector%s set up.\n' % (nsetupable,['y','ies'][nsetupable!=1])
    if nsetupable==0:
      continue
    break


  # select range within initconds file
  # only start index needed, end index is determined by number of trajectories
  print 'Please enter the index of the first initial condition in the initconds file to be setup.'
  while True:
    firstindex=question('Starting index:',int,[1])[0]
    if not 0<firstindex<=INFOS['ninit']:
      print 'Please enter an integer between %i and %i.' % (1,INFOS['ninit'])
      continue
    nsetupable=0
    for i,initcond in enumerate(INFOS['initlist']):
      if i+1<firstindex:
        continue
      for state in set(setupstates):
        try:
          nsetupable+=initcond.statelist[state-1].Excited
        except IndexError:
          break
    print '\nThere can be %i trajector%s set up, starting in %i states.' % (nsetupable,['y','ies'][nsetupable!=1],len(INFOS['setupstates']))
    if nsetupable==0:
      continue
    break
  INFOS['firstindex']=firstindex


  # Number of trajectories
  print '\nPlease enter the total number of trajectories to setup.'
  while True:
    ntraj=question('Number of trajectories:',int,[nsetupable])[0]
    if not 1<=ntraj<=nsetupable:
      print 'Please enter an integer between %i and %i.' % (1,nsetupable)
      continue
    break
  INFOS['ntraj']=ntraj


  # Random number seed
  print '\nPlease enter a random number generator seed (type "!" to initialize the RNG from the system time).'
  while True:
    line=question('RNG Seed: ',str,'!',False)
    if line=='!':
      random.seed()
      break
    try:
      rngseed=int(line)
      random.seed(rngseed)
    except ValueError:
      print 'Please enter an integer or "!".'
      continue
    break
  print ''



  INFOS['interface']=int(8)

  INFOS['needed']=[]

  INFOS['nn_trainingsdata']=False


  # Dynamics options
  string='\n  '+'='*80+'\n'
  string+='||'+centerstring('Surface Hopping dynamics settings',80)+'||\n'
  string+='  '+'='*80+'\n\n'
  print string


  # Simulation time
  print centerstring('Simulation time',60,'-')+'\n'
  print 'Please enter the total simulation time.'
  while True:
    num=question('Simulation time (fs):',float,[1000.])[0]
    if num<=0:
      print 'Simulation time must be positive!'
      continue
    break
  INFOS['tmax']=num


  # Timestep
  print '\nPlease enter the simulation timestep (0.5 fs recommended).'
  while True:
    dt=question('Simulation timestep (fs):',float,[0.5])[0]
    if dt<=0:
      print 'Simulation timestep must be positive!'
      continue
    break
  INFOS['dtstep']=dt
  print '\nSimulation will have %i timesteps.' % (num/dt+1)


  # number of substeps
  print '\nPlease enter the number of substeps for propagation (25 recommended).'
  while True:
    nsubstep=question('Nsubsteps:',int,[25])[0]
    if nsubstep<=0:
      print 'Enter a positive integer!'
      continue
    break
  INFOS['nsubstep']=nsubstep


  # whether to kill relaxed trajectories
  print '\nThe trajectories can be prematurely terminated after they run for a certain time in the lowest state. '
  INFOS['kill']=question('Do you want to prematurely terminate trajectories?',bool,False)
  if INFOS['kill']:
    while True:
      tkill=question('Kill after (fs):',float,[10.])[0]
      if tkill<=0:
        print 'Must be positive!'
        continue
      break
    INFOS['killafter']=tkill
  print ''


  print '\n'+centerstring('Dynamics settings',60,'-')


  # SHARC or FISH
  print '\nDo you want to perform the dynamics in the diagonal representation (SHARC dynamics) or in the MCH representation (regular surface hopping)?'
  surf=question('SHARC dynamics?',bool,True)
  INFOS['surf']=['mch','diagonal'][surf]

  ## SOC or not
  recommended=True
  #if len(INFOS['states'])==1:
    #recommended=False
  print '\nDo you want to include spin-orbit couplings in the dynamics?'
  INFOS['soc']=question('Spin-orbit couplings?',bool,recommended)

  # Setup SOCs
  if len(INFOS['states'])>1:
      print 'Do you want to include spin-orbit couplings in the dynamics?\n'
      soc=question('Spin-Orbit calculation?',bool,True)
      if soc:
        print 'Will calculate spin-orbit matrix.'
  else:
    print 'Only singlets specified: not calculating spin-orbit matrix.'
    soc=False
  print ''
  INFOS['states']=states
  INFOS['nstates']=nstates
  INFOS['soc']=soc
  # Coupling
  print '\nPlease choose the quantities to describe non-adiabatic effects between the states:'
  for i in Couplings:
    print '%i\t%s' % (i,Couplings[i]['description'])
  print ''
  default=None
  num=question('Coupling number:',int,default)[0]
  INFOS['coupling']=num


  # Phase tracking
  INFOS['phases_from_interface']=False

  # Gradient correction (only for SHARC)
  if INFOS['surf']=='diagonal':
    recommended=Couplings[INFOS['coupling']]['name']=='nacdr'
    print '\nFor SHARC dynamics, the evaluation of the mixed gradients necessitates to calculate non-adiabatic coupling vectors %s.' % (['(Extra computational cost)',' (Recommended)'][recommended])
    INFOS['gradcorrect']=question('Include non-adiabatic couplings in the gradient transformation?',bool,recommended)
  else:
    INFOS['gradcorrect']=False


  # Kinetic energy modification
  print '\nDuring a surface hop, the kinetic energy has to be modified in order to conserve total energy. There are several options to that:'
  cando=[]
  for i in EkinCorrect:
    recommended=len(EkinCorrect[i]['required'])==0  or  Couplings[INFOS['coupling']]['name'] in EkinCorrect[i]['required']
    cando.append(i)
    print '%i\t%s%s' % (i, EkinCorrect[i]['description'],['\n\t(extra computational cost)',''][ recommended ])
  while True:
    ekinc=question('EkinCorrect:',int,[2])[0]
    if ekinc in EkinCorrect and ekinc in cando:
      break
    else:
      print 'Please input one of the following: %s!' % ([i for i in cando])
  INFOS['ekincorrect']=ekinc


  # frustrated reflection
  print '\nIf a surface hop is refused (frustrated) due to insufficient energy, the velocity can either be left unchanged or reflected:'
  cando=[]
  for i in EkinCorrect:
    recommended=len(EkinCorrect[i]['required'])==0  or  Couplings[INFOS['coupling']]['name'] in EkinCorrect[i]['required']
  while True:
    reflect=question('Reflect frustrated:',int,[1])[0]
    break
  INFOS['reflect']=reflect


  # decoherence
  print '\nPlease choose a decoherence correction for the %s states:' % (['MCH','diagonal'][INFOS['surf']=='diagonal'])
  cando=[]
  for i in Decoherences:
    recommended=len(Decoherences[i]['required'])==0  or  Couplings[INFOS['coupling']]['name'] in Decoherences[i]['required']
    print '%i\t%s' % (i, Decoherences[i]['description'] )
  while True:
    decoh=question('Decoherence scheme:',int,[2])[0]
    break
  INFOS['decoherence']=[Decoherences[decoh]['name'],Decoherences[decoh]['params']]

  # surface hopping scheme
  print '\nPlease choose a surface hopping scheme for the %s states:' % (['MCH','diagonal'][INFOS['surf']=='diagonal'])
  cando=list(HoppingSchemes)
  for i in HoppingSchemes:
    print '%i\t%s' % (i, HoppingSchemes[i]['description'])
  while True:
    hopping=question('Hopping scheme:',int,[2])[0]
    if hopping in HoppingSchemes and hopping in cando:
      break
    else:
      print 'Please input one of the following: %s!' % ([i for i in cando])
  INFOS['hopping']=HoppingSchemes[hopping]['name']


  # Scaling
  print '\nDo you want to scale the energies and gradients?'
  scal=question('Scaling?',bool,False)
  if scal:
    while True:
      fscal=question('Scaling factor (>0.0): ',float)[0]
      if fscal<=0:
        print 'Please enter a positive real number!'
        continue
      break
    INFOS['scaling']=fscal
  else:
    INFOS['scaling']=False


  # Damping
  print '\nDo you want to damp the dynamics (Kinetic energy is reduced at each timestep by a factor)?'
  damp=question('Damping?',bool,False)
  if damp:
    while True:
      fdamp=question('Scaling factor (0-1): ',float)[0]
      if not 0<=fdamp<=1:
        print 'Please enter a real number 0<=r<=1!'
        continue
      break
    INFOS['damping']=fdamp
  else:
    INFOS['damping']=False


  # atommask
  INFOS['atommaskarray']=[]
  if (INFOS['decoherence'][0]=='edc') or (INFOS['ekincorrect']==2) or (INFOS['reflect']==2):
    print '\nDo you want to use an atom mask for velocity rescaling or decoherence?'
    if question('Atom masking?',bool,False):
      print '\nPlease enter all atom indices (start counting at 1) of the atoms which should be masked. \nRemember that you can also enter ranges (e.g., "-1~-3  5  11~21").'
      arr=question('Masked atoms:',int,ranges=True)
      for i in arr:
        if 1<=i<=INFOS['natom']:
          INFOS['atommaskarray'].append(i)

  # selection of gradients (only for SHARC) and NACs (only if NAC=ddr)
  print '\n'+centerstring('Selection of Gradients and NACs',60,'-')+'\n'
  print '''In order to speed up calculations, SHARC is able to select which gradients and NAC vectors it has to calculate at a certain timestep. The selection is based on the energy difference between the state under consideration and the classical occupied state.
'''
  if INFOS['surf']=='diagonal':
    if INFOS['soc']:
      sel_g=question('Select gradients?',bool,False)
    else:
      sel_g=True
  else:
    sel_g=False
  INFOS['sel_g']=sel_g
  if Couplings[INFOS['coupling']]['name']=='ddr' or INFOS['gradcorrect'] or EkinCorrect[INFOS['ekincorrect']]['name']=='parallel_nac':
    sel_t=question('Select non-adiabatic couplings?',bool,False)
  else:
    sel_t=False
  INFOS['sel_t']=sel_t
  if sel_g or sel_t:
    if not sel_t and not INFOS['soc']:
      INFOS['eselect']=0.001
      print '\nSHARC dynamics without SOC and NAC: setting minimal selection threshold.'
    else:
      print '\nPlease enter the energy difference threshold for the selection of gradients and non-adiabatic couplings (in eV). (0.5 eV recommended, or even larger if SOC is strong in this system.)'
      eselect=question('Selection threshold (eV):',float,[0.5])[0]
      INFOS['eselect']=abs(eselect)


  # Interface-specific section
  INFOS=get_SchNet(INFOS)


  # Dynamics options
  string='\n  '+'='*80+'\n'
  string+='||'+centerstring('Content of output.dat files',80)+'||\n'
  string+='  '+'='*80+'\n'
  print string
  
  #options for writing QM.in and QM.out
  print '\n Do you want to skip the writing of a QM.in and QM.out file each timestep?'
  write_qminqmout=question('Write QM.in and QM.out?',bool,True)
  if write_qminqmout:
    INFOS['write_qminqmout']=False
  else:
    INFOS['write_qminqmout']=True
  # options for writing to output.dat
  print '\nDo you want to write the gradients to the output.dat file ?'
  write_grad=question('Write gradients?',bool,False)
  if write_grad:
    INFOS['write_grad']=True
  else:
    INFOS['write_grad']=False
    
  print '\nDo you want to write the non-adiabatic couplings (NACs) to the output.dat file ?'
  write_NAC=question('Write NACs?',bool,False)
  if write_NAC==True:
    INFOS['write_NAC']=True
  else:
    INFOS['write_NAC']=False


  print '\nDo you want to write property matrices to the output.dat file  (e.g., Dyson norms)?'
  if 'ion' in INFOS and INFOS['ion']:
    INFOS['write_property2d']=question('Write property matrices?',bool,True)
  else:
    INFOS['write_property2d']=question('Write property matrices?',bool,False)


  print '\nDo you want to write property vectors to the output.dat file  (e.g., TheoDORE results)?'
  if 'theodore' in INFOS and INFOS['theodore']:
    INFOS['write_property1d']=question('Write property vectors?',bool,True)
  else:
    INFOS['write_property1d']=question('Write property vectors?',bool,False)


  print '\nDo you want to write the overlap matrix to the output.dat file ?'
  INFOS['write_overlap']=question('Write overlap matrix?',bool, (Couplings[INFOS['coupling']]['name']=='overlap') )

  # Add some simple keys
  INFOS['printlevel']=2
  INFOS['cwd']=os.getcwd()
  print ''
  return INFOS


# ======================================================================================================================   
def get_SchNet(INFOS):

    INFOS['script']='/schnarc_md.py'
    cwdpath=os.getcwd()
    INFOS['cwd']=cwdpath
    INFOS['cwdNN']=cwdpath+'/NN'
    #NN executable
    print centerstring('Path to SchNarc',60,'-')+'\n'
    path=os.getenv('SCHNARC')
    path=os.path.expanduser(os.path.expandvars(path))
    if not path=='':
      path='$SCHNARC/'
    else:
      path=None
    print '\nPlease specify path to SchNarc directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n'
    pathNN=question('Path to NN executable:',str,path)
    pathNN=os.path.expanduser(os.path.expandvars(pathNN))
    INFOS['NN']=pathNN
    print ''
    #SHARC executable
    print centerstring('Path to SHARC',60,'-')+'\n'
    path=os.getenv('SHARC')
    path=os.path.expanduser(os.path.expandvars(path))
    if not path=='':
      path='$SHARC'
    else:
      path=None
    print '\nPlease specify path to SHARC directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n'
    pathNN=question('Path to SHARC executable:',str,path)
    pathNN=os.path.expanduser(os.path.expandvars(pathNN))
    INFOS['SHARCpath']=pathNN
    print ''
    print centerstring('Path to the training data base',60,'-')+'\n'
    print '\nPlease specify the path to the data base used for training of neural networks.\n'
    datapath=question('Training data base:',str)
    datapath=os.path.expandvars(os.path.expanduser(datapath))
    INFOS['datapath']=datapath
    print centerstring('Path to trained SchNet model',60,'-')+'\n'
    modelpath=question('Please specify the path to the SchNet model',str)[0:]
    INFOS['modelpath']=modelpath
    #adaptive sampling 
    print centerstring('Adaptive Sampling',60,'-')+'\n'
    Adaptive=question('Do you want to carry out adaptive sampling',bool,False)
    if Adaptive:
      INFOS['adaptive'] = "--adaptive 1.0"
      pathNN2=question('Please specify the path to the second trained SchNet model',str)[0:]
      INFOS['modelpath2'] = pathNN2
      INFOS['adaptive_model'] = "--modelpaths"
      threshold_E=question('Specify the threshold for energies in a.u.',float)[0]
      threshold_F=question('Specify the threshold for forcess in a.u.',float)[0]
      threshold_Mu=question('Specify the threshold for dipoles in a.u.',float)[0]
      threshold_NAC=question('Specify the threshold for nacs in a.u.',float)[0]
      threshold_SOC=question('Specify the threshold for socs in a.u.',float)[0]
      INFOS['adaptive_thresholds'] = "--thresholds %s %s %s %s %s " %(threshold_E,threshold_F,threshold_Mu,threshold_NAC,threshold_SOC)
      printuncertainty = question('Do you want to print the uncertainty between the networks?',bool,False)
      if printuncertainty == True:
          INFOS['print_uncertainty'] = "--print_uncertainty"
      else:
          INFOS['print_uncertainty'] = ""
    else:
      INFOS['adaptive'] = ""
      INFOS['modelpath2'] = ""
      INFOS['adaptive_model'] = ""
      INFOS['adaptive_thresholds'] = ""
      INFOS['print_uncertainty'] = ""
    #get options for the second interface used to generate trainingsdata
    #NAC approximation 
    while True:
      print centerstring('NN-NAC-Approximation',60,'-')+'\n'
      NACapprox=question('Do you want to approximate NAC vectors by using the direction from difference-Hessians and the magnitude from energy gaps?',bool,True)
      if NACapprox:
        INFOS['NACapprox'] = "--hessian --nac_approx"
        deltaH=question('Which NAC approximation do you want to use? Set 1 for an accurate ML model and 2 for a model that slightly overestimates energy gaps',float)[0]
        threshold_dE_S=question('Which energy gap do you want to use for computation of singlet-singlet couplings? 0.02H (0.5eV) is the default.',float)[0]
        threshold_dE_T=question('Which energy gap do you want to use for computation of triplet-triplet couplings? 0.02H (0.5eV) is the default.',float)[0]
      else:
        INFOS['NACapprox'] = ""
      INFOS['deltaH']=""
      INFOS['threshold_dE_S']=""
      INFOS['threshold_dE_T']=""
      break

    num=float(0.1)
    dt=float(0.5)
    print '\nSimulation will have %i timesteps.' % (num/dt+1)


    string='\n  '+'='*80+'\n'
    string+='||'+centerstring('Surface Hopping dynamics settings',80)+'||\n'
    string+='  '+'='*80+'\n\n'
    # Interface-specific section
    return INFOS


def copyinputfile_NN(INFOS,iconddir):
  # copy optionsfile
  return
def copy_weightfile_NN(INFOS,iconddir):
  # copy optionsfile
  return

def copysh2file_NN(INFOS,iconddir):
  return

#=======================================================================================================================================#
#=======================================================================================================================================#
#=======================================================================================================================================#


def get_runscript_info(INFOS):
  ''''''

  string='\n  '+'='*80+'\n'
  string+='||'+centerstring('Run mode setup',80)+'||\n'
  string+='  '+'='*80+'\n\n'
  print string

  print centerstring('Run script',60,'-')+'\n'
  print '''This script can generate the run scripts for each trajectory in two modes:

  - In the first mode, the calculation is run in subdirectories of the current directory.

  - In the second mode, the input files are transferred to another directory (e.g. a local scratch directory), the calculation is run there, results are copied back and the temporary directory is deleted. Note that this temporary directory is not the same as the scratchdir employed by the interfaces.

Note that in any case this script will setup the input subdirectories in the current working directory. 
'''
  print '(actually perform the calculations in subdirectories of: %s)\n' % (INFOS['cwd'])
  here=question('Calculate here?',bool,False)
  if here:
    INFOS['here']=True
    INFOS['copydir']=INFOS['cwd']
    if INFOS['interface']==8:
      INFOS['copyQMin']=INFOS['cwd']
  else:
    INFOS['here']=False
    print '\nWhere do you want to perform the calculations? Note that this script cannot check whether the path is valid. NOTE: For NNs $TMPDIR is only possible for running dynamics without any trainings step. In any other case you have to specify your path!'
    INFOS['copydir']=question('Run directory?',str)
    if INFOS['interface']==8:
      INFOS['copyQMin']=INFOS['cwd']
  print ''

  print centerstring('Submission script',60,'-')+'\n'
  print '''During the setup, a script for running all initial conditions sequentially in batch mode is generated. Additionally, a queue submission script can be generated for all initial conditions.
'''
  qsub=question('Generate submission script?',bool,False)
  if not qsub:
    INFOS['qsub']=False
  else:
    INFOS['qsub']=True
    if INFOS['interface']==8:
      print '\nPlease enter a queue submission command, including possibly options to the queueing system,\ne.g. for SGE: "qsub -q queue.q -S /bin/bash -cwd" (Currently only polonium and lead possible for NN)'
    else:
      print '\nPlease enter a queue submission command, including possibly options to the queueing system,\ne.g. for SGE: "qsub -q queue.q -S /bin/bash -cwd" (Do not type quotes!).'
    INFOS['qsubcommand']=question('Submission command?',str,None,False)
    INFOS['proj']=question('Project Name:',str,None,False)


  print ''
  return INFOS

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def make_directory(iconddir):
  '''Creates a directory'''

  if os.path.isfile(iconddir):
    print '\nWARNING: %s is a file!' % (iconddir)
    return -1
  if os.path.isdir(iconddir):
    if len(os.listdir(iconddir))==0:
      return 0
    else:
      print '\nWARNING: %s/ is not empty!' % (iconddir)
      if not 'overwrite' in globals():
        global overwrite
        overwrite=question('Do you want to overwrite files in this and all following directories? ',bool,False)
      if overwrite:
        return 0
      else:
        return -1
  else:
    try:
      os.mkdir(iconddir)
    except OSError:
      print '\nWARNING: %s cannot be created!' % (iconddir)
      return -1
    return 0

# ======================================================================================================================

def writeSHARCinput(INFOS,initobject,iconddir,istate):

  inputfname=iconddir+'/input'
  try:
    inputf=open(inputfname, 'w')
  except IOError:
    print 'IOError during writeSHARCinput, iconddir=%s\n%s' % (iconddir,inputfname)
    quit(1)

  s='printlevel 2\n\ngeomfile "geom"\nveloc external\nvelocfile "veloc"\n\n'
  s+='nstates '
  for nst in INFOS['states']:
    s+='%i ' % nst
  s+='\nactstates '
  for nst in INFOS['actstates']:
    s+='%i ' % nst
  s+='\nstate %i %s\n' % (istate,['mch','diag'][INFOS['diag']])
  s+='coeff auto\n'
  s+='rngseed %i\n\n' % (random.randint(-32768,32767))
  s+='ezero %18.10f\n' % (INFOS['eref'])

  s+='tmax %f\nstepsize %f\nnsubsteps %i\n' % (INFOS['tmax'],INFOS['dtstep'],INFOS['nsubstep'])
  if INFOS['kill']:
    s+='killafter %f\n' % (INFOS['killafter'])
  s+='\n'

  s+='surf %s\n' % (INFOS['surf'])
  s+='coupling %s\n' % (Couplings[INFOS['coupling']]['name'])
  s+='%sgradcorrect\n' % (['no',''][INFOS['gradcorrect']])
  s+='ekincorrect %s\n' % (EkinCorrect[INFOS['ekincorrect']]['name'])
  s+='reflect_frustrated %s\n' % (EkinCorrect[INFOS['reflect']]['name'])
  s+='decoherence_scheme %s\n' % (INFOS['decoherence'][0])
  if INFOS['decoherence'][1]:
    s+='decoherence_param %s\n' % (INFOS['decoherence'][1])
  s+='hopping_procedure %s\n' % (INFOS['hopping'])
  if INFOS['scaling']:
    s+='scaling %f\n' % (INFOS['scaling'])
  if INFOS['damping']:
    s+='dampeddyn %f\n' % (INFOS['damping'])
  if INFOS['sel_g']:
    s+='grad_select\n'
  else:
    s+='grad_all\n'
  if INFOS['sel_t']:
    s+='nac_select\n'
  else:
    if Couplings[INFOS['coupling']]['name']=='ddr' or INFOS['gradcorrect'] or EkinCorrect[INFOS['ekincorrect']]['name']=='parallel_nac':
      s+='nac_all\n'
  if 'eselect' in INFOS:
    s+='eselect %f\n' % (INFOS['eselect'])
    s+='select_directly\n'
  if INFOS['soc']:
    s+='nospinorbit\n'
  #if not INFOS['socs']:
  #  s+='nospinorbit\n'

  if INFOS['write_grad']:
    s+='write_grad\n'
  if INFOS['write_NAC']==True:
    s+='write_nacdr\n'
  if INFOS['write_overlap']:
    s+='write_overlap\n'
  if INFOS['write_property1d']:
    s+='write_property1d\n'
    s+='n_property1d %i\n' % (INFOS['theodore.count'])
  if INFOS['write_property2d']:
    s+='write_property2d\n'
    s+='n_property2d %i\n' % (1)


  if 'ion' in INFOS and INFOS['ion']:
    s+='ionization\n'
    s+='ionization_step 1\n'

  if 'theodore' in INFOS and INFOS['theodore']:
    s+='theodore\n'
    s+='theodore_step 1\n'

  inputf.write(s)
  inputf.close()

  # geometry file
  geomfname=iconddir+'/geom'
  geomf=open(geomfname,'w')
  for atom in initobject.atomlist:
    geomf.write(atom.geomstring()+'\n')
  geomf.close()

  # velocity file
  velocfname=iconddir+'/veloc'
  velocf=open(velocfname,'w')
  for atom in initobject.atomlist:
    velocf.write(atom.velocstring()+'\n')
  velocf.close()


  return

# ======================================================================================================================

def writeRunscript(INFOS,iconddir):
  '''writes the runscript in each subdirectory'''
  try:
    # if NNs are used, the runscript will be written into the subdirectory QM
    if INFOS['interface']==8:
      runscript=open('%s/run.sh' % ('NN/'+iconddir), 'w')
    #this is the option for every interface except for NNs
    else:
      runscript=open('%s/run.sh' % (iconddir), 'w' )
  except IOError:
    print 'IOError during writeRunscript, iconddir=%s' % (iconddir)
    quit(1)
  if 'proj' in INFOS:
    projname='%4s_%5s' % (INFOS['proj'][0:4],iconddir[-6:-1])
  else:
    projname='traj_%5s' % (iconddir[-6:-1])

  # ================================ 
  intstring=''
  if 'adfrc' in INFOS:
    intstring='. %s' % (INFOS['adfrc'])

  # ================================ for here mode
  if INFOS['interface']==8:
        string='''#$-N %s
PRIMARY_DIR=%s/%s
SCHNARC=%s
SHARC=%s
. ~/.bashrc


cd $PRIMARY_DIR/
printf '' > 'RUN'
rm -f STOP
python -O  $SCHNARC/%s pred %s %s %s %s %s %s %s %s %s %s %s >> NN.log 2>> NN.err
grep -q -F "restart" input || echo "restart" >> input
rm -f STOP

''' % (projname,INFOS['cwdNN'],iconddir,INFOS['NN'],INFOS['SHARCpath'],INFOS['script'],INFOS['datapath'],INFOS['modelpath'],INFOS['NACapprox'],INFOS['deltaH'],INFOS['threshold_dE_S'],INFOS['threshold_dE_T'],INFOS['adaptive'],INFOS['adaptive_model'],INFOS['modelpath2'],INFOS['adaptive_thresholds'],INFOS['print_uncertainty'])
  if INFOS['qsub']:
    string='#$ -v USER_EPILOG=%s/epilog.sh' % (iconddir)
    if INFOS['interface']==8:
        # sed -i "/End of header array data/q" output.dat
        #copy and primary dir with index 2 refer to the QM calcualtions
        if INFOS['write_qminqmout']==False:
          string='''#$-N %s

PRIMARY_DIR=%s/%s
COPY_DIR=%s
SCHNARC=%s
SHARC=%s
. ~/.bashrc

cp -r $PRIMARY_DIR/* $COPY_DIR/

cd $COPY_DIR/

l=1
j=0

python  $SCHNARC/%s pred %s %s %s %s %s %s %s %s %s %s %s >> NN.log 2>> NN.err

mv -r $COPY_DIR/* $PRIMARY_DIR/
mv $COPY_DIR/* $PRIMARY_DIR/
exit

''' % (projname,INFOS['cwdNN'],iconddir,INFOS['copydir'],INFOS['NN'],INFOS['SHARCpath'],INFOS['script'],INFOS['datapath'],INFOS['modelpath'],INFOS['NACapprox'],INFOS['deltaH'],INFOS['threshold_dE_S'],INFOS['threshold_dE_T'],INFOS['adaptive'],INFOS['adaptive_model'],INFOS['modelpath2'],INFOS['adaptive_thresholds'],INFOS['print_uncertainty'])
        else:
          INFOS['script']='/schnarc_md.py'
          string='''#$-N %s

PRIMARY_DIR=%s/%s
COPY_DIR=%s
SCHNARC=%s
SHARC=%s
. ~/.bashrc

mkdir $COPY_DIR
cp -r $PRIMARY_DIR/* $COPY_DIR/

cd $COPY_DIR/
printf '' > 'RUN'

python $SCHNARC/%s pred %s %s %s %s %s %s %s %s %s %s %s >> NN.log 2>> NN.err
mv -r $COPY_DIR/* $PRIMARY_DIR/
mv $COPY_DIR/* $PRIMARY_DIR/
exit


''' % (projname,INFOS['cwdNN'],iconddir,INFOS['copydir'],INFOS['NN'],INFOS['SHARCpath'],INFOS['script'],INFOS['datapath'],INFOS['modelpath'],INFOS['NACapprox'],INFOS['deltaH'],INFOS['threshold_dE_S'],INFOS['threshold_dE_T'],INFOS['adaptive'],INFOS['adaptive_model'],INFOS['modelpath2'],INFOS['adaptive_thresholds'],INFOS['print_uncertainty'])

  runscript.write(string)
  runscript.close()

  if INFOS['interface']==8:
      filename='NN/'+iconddir+'/run.sh'
      os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR)
  else:
    filename=iconddir+'/run.sh'
    os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR)

  return


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def get_iconddir(istate,INFOS):
  if INFOS['diag']:
    dirname='State_%i' % (istate)
  else:
    mult,state,ms=INFOS['statemap'][istate]
    dirname=IToMult[mult]+'_%i' % (state-(mult==1 or mult==2))
  return dirname

# ====================================

def setup_all(INFOS):
    '''This routine sets up the directories for the initial calculations.'''

    string='\n  '+'='*80+'\n'
    string+='||'+centerstring('Setting up directories...',80)+'||\n'
    string+='  '+'='*80+'\n\n'
    print string

    INFOS['copydirNN']=INFOS['copydir']
    #make subdirectories NN (dynamics) and QM (training data generation) for NNs
    io=make_directory(INFOS['cwdNN'])
    if io!=0:
      print 'Could not make directory %s' % (INFOS['cwdNN'])
      quit(1)


    for istate in INFOS['setupstates']:
      dirname=get_iconddir(istate,INFOS)
      io=make_directory('NN/'+dirname)
      if io!=0:
        print 'Could not make directory %s' % (dirname)
        quit(1)
      if io!=0:
        print 'Could not make directory %s' % (dirname)
        quit(1)
     
      width=50
      ntraj=INFOS['ntraj']
      idone=0
      finished=False
      initlist=INFOS['initlist']
    for icond in range(INFOS['firstindex'],INFOS['ninit']+1):
      for istate in INFOS['setupstates']:

        if len(initlist[icond-1].statelist)<istate:
          continue
        if not initlist[icond-1].statelist[istate-1].Excited:
          continue
    
        idone+=1
        done=idone*width/ntraj
        sys.stdout.write('\rProgress: ['+'='*done+' '*(width-done)+'] %3i%%' % (done*100/width))
        dirname=get_iconddir(istate,INFOS)+'/TRAJ_%05i/' % (icond)
        print(dirname)
        #print dirnameNN, dirnameQM
        io=make_directory('NN/'+dirname)
        if io!=0:
          print 'Skipping initial condition %i %i!' % (istate, icond)
          continue
        if io!=0:
          print 'Skipping initial condition %i %i!' % (istate, icond)
          #writes input file ones for NN and ones for QM with different tmax, dt,...
        writeSHARCinput(INFOS,initlist[icond-1],'NN/'+dirname,istate)
        #creates directories in both subdirectories QM and NN
        io=make_directory('NN/'+dirname+'/QM')
        io+=make_directory('NN/'+dirname+'/restart')
        if io!=0:
          print 'Could not make QM or restart directory!'
          continue
        #change copydir and cwd to prepare NN directory
        INFOS['cwd']=INFOS['cwdNN']
        INFOS['copydir']=INFOS['copydirNN']
        writeRunscript(INFOS,dirname)
        gen_QMin(INFOS,'NN/'+dirname,icond)
        if idone==ntraj:
          finished=True
          break 
      if finished:
        print '\n\n%i trajectories setup, last initial condition was %i in state %i.\n' % (ntraj,icond,istate)
        setup_stat=open('setup_traj.status','a')
        string='''*** %s %s %s
      First index:          %i
      Last index:           %i
      Trajectories:         %i
      State of last traj.:  %i

    '''   % (datetime.datetime.now(),
           gethostname(),
           os.getcwd(),
           INFOS['firstindex'],
           icond,
           ntraj,
           istate)
        setup_stat.write(string)
        setup_stat.close()
        break

    print '\n'


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def main():
  '''Main routine'''

  usage='''
python setup_traj.py

This interactive program prepares SHARC dynamics calculations.
'''

  description=''
  parser = OptionParser(usage=usage, description=description)

  displaywelcome()
  open_keystrokes()

  INFOS=get_general()

  INFOS=get_runscript_info(INFOS)
  """print '\n'+centerstring('Full input',60,'#')+'\n'
  for item in INFOS:
    if not 'initlist' in item:
      print item, ' '*(25-len(item)), INFOS[item]
  print ''"""
  setup=question('Do you want to setup the specified calculations?',bool,True)
  print ''

  if setup:
    setup_all(INFOS)

  close_keystrokes()



# ======================================================================================================================

if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    print '\nCtrl+C makes me a sad SchNarc ;-(\n'
    quit(0)
