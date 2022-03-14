import argparse
import schnetpack as spk
import torch
from torch.utils.data.sampler import RandomSampler
from torch.optim import Adam
import os
import numpy as np
import shutil
from schnarc.schnarc import get_energy, get_gradient, get_dipoles, get_nacs, get_schnarc, get_socs, get_diab,get_nacs_deltaH, get_nacs_deltaH2, get_nacs_deltaH3, get_nacs_deltaH4

def QMout(prediction,modelpath,nac_approx,n_states):
    #returns predictions in QM.out format useable with SHARC
    QMout_string=''
    QMout_energy=''
    QMout_gradient=''
    QMout_dipoles=''
    QMout_nacs=''
    if int(prediction['energy'].shape[0]) == int(1):
        for property in prediction.keys():
            if property == "energy":
                QMout_energy=get_energy(prediction['energy'][0],n_states)
                QMout_gradient=get_gradient(prediction['gradients'][0],n_states,prediction['energy'][0])
            #elif property == "diab":
            #    QMout_energy=get_diab(prediction['diab2'][0],n_states)
            elif property == "dipoles":
                QMout_dipoles=get_dipoles(prediction['dipoles'][0],n_states,prediction['energy'][0])
            elif property == "nacs":
                QMout_nacs=get_nacs(prediction['nacs'][0],n_states)
            elif property == "socs":
                QMout_energy=get_socs(prediction['socs'][index],n_states,prediction['energy'][index])
        QM_out = open("%s/QM.out" %modelpath, "w")
        QMout_string=QMout_energy+QMout_dipoles+QMout_gradient+QMout_nacs
        QM_out.write(QMout_string)
        QM_out.close()

    else:
        for index in range(prediction['energy'].shape[0]):
            os.system('mkdir %s/Geom_%04d' %(modelpath,index+1))
            for property in prediction.keys():
                if property == "energy":
                    QMout_energy=get_energy(prediction['energy'][index],n_states)
                elif property == "diab":
                    pass
                    #QMout_energy=get_diab(prediction['diab2'][index],prediction['energy'][index])
                elif property == "socs":
                    QMout_energy=get_socs(prediction['socs'][index],n_states,prediction['energy'][index])
                elif property == "gradients":
                    QMout_gradient=get_gradient(prediction['gradients'][index],n_states,prediction['energy'][index])
                elif property == "dipoles":
                    QMout_dipoles=get_dipoles(prediction['dipoles'][index],n_states,prediction['energy'][index])
                elif property == "nacs":
                        QMout_nacs=get_nacs(prediction['nacs'][index],n_states)
            if nac_approx==int(1):
                        QMout_nacs = get_nacs_deltaH(prediction['hessian'][index],prediction['energy'][index],prediction['gradients'][index],n_states)
            elif nac_approx==int(2):
                        QMout_nacs = get_nacs_deltaH2(prediction['hessian'][index],prediction['energy'][index],prediction['gradients'][index],n_states)
            elif nac_approx==int(3):
                        QMout_nacs = get_nacs_deltaH3(prediction['hessian'][index],prediction['energy'][index],prediction['gradients'][index],n_states)
            elif nac_approx==int(4):
                        QMout_nacs = get_nacs_deltaH4(prediction['hessian'][index],prediction['energy'][index],prediction['gradients'][index],n_states)
            else:
              pass
            QM_out = open("QM.out", "w")
            QMout_string=QMout_energy+QMout_dipoles+QMout_gradient+QMout_nacs
            QM_out.write(QMout_string)
            QM_out.close()
            os.system("mv QM.out %s/Geom_%04d/" %(modelpath,index+1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('database', type=str, help='Path to database')
    parser.add_argument('modelpath', type=str, help='Path to database')
    parser.add_argument('--split', type=int, nargs=2, help='Split')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--cuda', action='store_true', help='Turn on cuda')
    parser.add_argument('--max_epochs', type=int, help='Maximum number of training epochs (default: %(default)s)',
                        default=5000)
    parser.add_argument('--lr', type=float, help='Initial learning rate (default: %(default)s)',
                        default=1e-4)
    parser.add_argument('--lr_patience', type=int,
                        help='Epochs without improvement before reducing the learning rate (default: %(default)s)',
                        default=25)
    parser.add_argument('--lr_decay', type=float, help='Learning rate decay (default: %(default)s)',
                        default=0.5)
    parser.add_argument('--lr_min', type=float, help='Minimal learning rate (default: %(default)s)',
                        default=1e-6)
    parser.add_argument('--rho', type=float,
                        help='Energy-gradient trade-off. For rho=0, use gradients only. (default: %(default)s)',
                        default=0.9)
    parser.add_argument('--rho_dipole',type=float,
                        help='Weighing factor of dipoles. For rho_dipole=1, properties are equally weighted. (default: %(default)s)',
                        default=0.001)
    parser.add_argument('--rho_nac',type=float,
                        help='Weighing factor of dipoles. For rho_nac=1, properties are equally weighted. (default: %(default)s)',
                        default=0.001)
    parser.add_argument('--evaluate', action='store_true', help='Run eval mode.')
    parser.add_argument('--logger', help='Choose logger for training process (default: %(default)s)',
                        choices=['csv', 'tensorboard'], default='csv')
    parser.add_argument('--log_every_n_epochs', type=int,
                        help='Log metrics every given number of epochs (default: %(default)s)',
                        default=1)
    choices = ['energy', 'gradient', 'dipoles', 'nacs']
    parser.add_argument('--properties', type=str, help="Possible properties: energy, gradient, dipoles, nacs", default=choices)
    parser.add_argument('--overwrite', action='store_true', help='Overwrite old directories')
    parser.add_argument('--n_features', default=256, type=int, help='Number of features used by SchNet.')
    parser.add_argument('--n_interactions', default=6, type=int, help='Number of interactions used by SchNet.')
    parser.add_argument('--return_QMout', action='store_true', help='print result in QM.out format of SHARC')
    parser.add_argument('--schnarc', action='store_true',help='run schnarc')
    parser.add_argument('--nac_approx',help="NAC approximation", type=int)
    args = parser.parse_args()

    # Get device
    device = torch.device("cuda" if args.cuda else "cpu")

    # Load data
    data = spk.data.AtomsData(args.database)
    n_states={}
    if data.get_metadata('n_triplets') == None:
        n_states['n_triplets']= 0
    else:
        n_states['n_triplets']=data.get_metadata("n_triplets")
    n_states['n_singlets']=data.get_metadata("n_singlets")
    n_states['states']=data.get_metadata("states")
    n_states['n_states'] = n_states['n_singlets']+3*n_states['n_triplets']
    if args.evaluate:
        prediction=np.load("%s/predictions.npz"%args.modelpath)
        os.system("mkdir %s/prediction" %args.modelpath)
        QMout(prediction,args.modelpath+"/prediction",args.nac_approx,n_states)
        exit()
