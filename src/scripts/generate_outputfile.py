# -*- coding: utf-8 -*-
#from sys import argv
import numpy as np
from glob import iglob
import shutil
import os
import readline
from sys import argv
#def readfile(filename):
 #   f=open(filename)
  #  out=f.readline()
   # f.close()
#    print "read"
 # except IOError:
  #  print 'File %s does not exist!' %/ (filename)
   # sys.exit(12)
 # return out
def write_outputfile(filename, stepsize):
  outputfile = open("output_all.dat", "w")
  header_exists=False
  for path, dirs, files in os.walk("."):
    #print files
    dirs.sort()
    for name in files:
      if name.endswith(filename):
        #infile=open(name)
        filepath=os.path.join(path, name)
        #print "h", filepath
        infile=open(filepath)
        f=infile.readlines()
        infile.close()
        is_header=True
        is_step=False
        step = 0
        for line in f:
          if not header_exists:
            outputfile.write(line)
          if 'End of header' in line:
            is_header=False
            header_exists=True
            continue
          if '! 0 Step' in line:
            if step < stepsize:
              #outputfile.write(line)
              step+=1 
            else:
              is_step=True
              continue
          if not is_header:
            if not is_step:
              outputfile.write(line)
 
  outputfile.close()


      #filestring=""
      #startlines=[]
      #iline=-1
      #print "Len", len(f)
      #while True:
        #iline+=1
        #if iline==len(f):
          #break
        #if "Step" in infile[iline]:
          #startlines.append(iline)
        #current=0
      #for iline in range(len(f)):
        #index=startlines[current]+3+iline
        #line = infile[index]
        #filestring+=line
      #print "done", filestring
      #string+=filestring
  #outputfile.write(string)
  #outputfile.close()

if __name__ == "__main__":
  try:
    name, filename, stepsize = argv
    stepsize = int( stepsize )
  except ValueError:
    print("Usage: script <filename (output.dat)> <stepsize>")
    exit()
  filename = argv[1]
  #filename="output.dat"
  #path="."
  #names=os.listdir(top)
  write_outputfile(filename, stepsize)
 # filename="output.dat"
#utfile.write( file1.read() )
#outfile.write( file2.read() )
