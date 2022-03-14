from sys import argv
from ase.io import read, write
from ase.db import connect
from ase.atoms import Atoms
import numpy as np


def parsedb(dbpath):
    db = connect(dbpath)
    # this function adds "has_forces" and "has_gradients" if this is not already stored in the db. 
    # it adds "dummy"-forces or "dummy"-gradients
    data={}
    geoms = read(dbpath,":")
    #check if gradients or forces are available
    keys=db.get(100).data.keys()
    print(keys)
    has_forces = False
    metadata=db.metadata
    has_gradients = False
    forces = False
    gradients = False
    has_gradients_always = False
    addkeys=[]
    has_forces_always = False
    if "has_forces" in keys:
        has_forces = True
    else:
        addkeys.append("has_forces")
    if "has_gradients" in keys:
        has_gradients = True
    else:
        addkeys.append("has_gradients")
    if "forces" in keys:
        forces = True
        if "has_forces" == False:
            has_forces_always = True
    else:
        addkeys.append("forces")
    if "gradients" in keys:
        gradients = True
        if "has_gradients" == False:
            has_gradients_always=True
    else:
        addkeys.append("gradients")
    return db,geoms,addkeys,has_forces_always, has_gradients_always,metadata



def adddata(db,addkeys,has_forces_always,has_gradients_always):
    data={}
    nstates = len(db.get(1).data["energy"])
    print(addkeys)
    print(db.get(1).data.keys())
    for i in range(len(db)):
        data[i]=db.get(i+1).data
        natoms = len(db.get(i+1).numbers)
        for key in addkeys:
            if key == "forces" or key == "gradients":
                data[i][key]=np.zeros((nstates,natoms,3))
            if key == "has_gradients":
                if has_gradients_always == True:
                    data[i][key]=1
                else:
                    data[i][key]=0
            if key=="has_forces":
                if has_forces_always==True:
                    data[i][key]=1
                else:
                    data[i][key]=0
    return data



def write_db(data,geoms,newdbname,metadata):
    newdb = connect(newdbname)
    for i in range(len(data)):
        newdb.write(geoms[i],data=data[i])
    newdb.metadata=metadata
    print("DB %s written."%newdbname)



if __name__ == '__main__':
    try:
        name, dbname, newdbname = argv
    except ValueError:
        print( "Usage: script <OldDBName> <NewDBName>")
  
    dbpath = str(argv[1])
    newdbname = str(argv[2])
    db,geoms,addkeys,has_forces_always, has_gradients_always,metadata = parsedb(dbpath)
    data=adddata(db,addkeys,has_forces_always, has_gradients_always)
    write_db(data,geoms,newdbname,metadata)

