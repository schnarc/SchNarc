import yaml
import logging

from schnet_ev.data import Properties


def generate_default_tradeoffs(yamlpath):
    tradeoffs = {p: 1.0 for p in Properties.properties}
    save_tradeoffs(tradeoffs, yamlpath)


def read_tradeoffs(yamlpath):
    with open(yamlpath, 'r') as tf:
        tradeoffs = yaml.load(tf)
    
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
