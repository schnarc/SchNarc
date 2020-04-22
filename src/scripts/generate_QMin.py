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

def question(question,typefunc,default=None,autocomplete=True):
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

    if typefunc==int or typefunc==float:
      # int and float will be returned as a list
      f=line.split()
      try:
        for i in range(len(f)):
          f[i]=typefunc(f[i])
        KEYSTROKES.write(line+' '*(40-len(line))+' #'+s+'\n')
        return f
      except ValueError:
        if typefunc==int:
          i=1
        elif typefunc==float:
          i=2
        print 'Please enter a %s' % ( ['string','integer','float'][i] )
        continue

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
	  #print line, "line"
          t=line.split()
          #string+='%s %s %s %s\n' %(t[0], t[2], t[3], t[4]) these values are given in bohr - but they are needed in angstrom
          string+='%s %s %s %s\n' %(t[0], float(t[2])*0.529177211, float(t[3])*0.529177211, float(t[4])*0.529177211)
          #print string
  string+='unit angstrom\n'
  string+='states '
  for i in range(len(states)):
    string+='%s ' %(states[i])
  string+='\ndt 0\nsavedir %s/%s\n' %(copydir,iconddir)
  string+='SOC\nDM\nGrad all\nNACdr\nphases'
  #print string
  QMin.write(string)
  QMin.close()
  #copy QM.in file to the necessary path, i.e. the State_x/TRAJ_xxxxx/QM folder
  #cpfrom='QM.in' %(copydir,iconddir)
  #cpto='%s/%sQM/QM.in' %(copydir,iconddir)
  #shutil.copy(cpfrom,cpto)
  #print cpfrom, cpto
  
  

if __name__ == "__main__":
    try:
        name, initcondsname = argv
    except ValueError:
        print "Usage: script <basename> <nsamples>"
        exit()
    INFOS={}
    initfile=question('Please specify the path to the initconds.excited file',False)
    INFOS['initcondsexcited']=initfile
    #not easy since the numbers of the TRAJ chosen are necessary!
