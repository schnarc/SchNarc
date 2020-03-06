from sys import argv
import numpy as np
import os

def extract_geometries(filename):
  #extract the xyz-geometries and writes it to an external file
  #the first line should contain the number of atoms
  initconds=open(filename,"r").readlines()
  
  geom=int(1)
  line_for_geom=int(0)
  for line in initconds: 
    #get number of atoms 
    data = line.split()
    atom_number=int(data[0])
    print(atom_number)
    break
  iline = -1 
  for line in initconds:
    iline+=1
    line_for_geom+=1
    if line_for_geom > int(atom_number)+2:
      line_for_geom=1
      geom+=1
      geomfile.close()
    geomfile=open("geom_%07d"%geom, "a")
    geomfile.write("%s" %line)
  return geom

def write_QMin(geom):
  numberofgeometries=geom
  #geom_middle is the initial geometry
  for i in range(1,numberofgeometries+1):
    os.system("mkdir Geom_%07d" %i)
    os.system("cp geom_%07d Geom_%07d/QM.in" %(i,i))
  #copy the second half of the geometries
    QM_in_1=open("Geom_%07d/QM.in" %i,"a")
    QM_in_1.close()
    os.system("cat QMstring >> Geom_%07d/QM.in" %i)
    os.system("rm -f geom_%07d" %i)
    #os.system("cp -r ICOND_00000/SAVE Geom_%07d_1/SAVE" %i)
    #os.system("cp -r ICOND_00000/SAVE Geom_%07d_2/SAVE" %i)

if __name__ == "__main__":
  #get name of input_file (currently "template.inp")
  try:
    name, input_file = argv
  except ValueError:
    print( "Usage: script <initconds.xyz>")
  filename = argv[1]
  geom=extract_geometries(filename)
  write_QMin(geom)
  exit()
