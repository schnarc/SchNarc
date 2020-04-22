import os
from sys import argv
def countstep(filename):
  textfile=open(filename, "r")
  inputString=textfile.readlines()
  i = 0
  for line in inputString:
    line = line.split()
    wanted='Step'
    if wanted in line:
      i+=1
  print( i)

if __name__ == "__main__":
  try:
    filename, outputfile = argv
  except ValueError:
    exit()
inputfile=argv[1]
countstep(inputfile)
