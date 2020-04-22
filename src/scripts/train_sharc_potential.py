import argparse
import schnetpack as spk
import torch
from torch.utils.data.sampler import RandomSampler
from torch.optim import Adam
import os
import shutil
from schnarc.schnarc import get_energy, get_force, get_dipoles, get_nacs, get_schnarc
from schnarc.model import MultiEnergy,MultiDipole,MultiNac,CombineProperties

def train(args, model, train_loader, val_loader, device, prop_dict):
    # setup hook and logging
    hooks = [
        spk.train.MaxEpochHook(args.max_epochs)
    ]

    # setup optimizer for training
    # to_opt = model.parameters()
    # Bugfix, since model will not train with requires grad variables
    to_opt = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(to_opt, lr=args.lr)

    schedule = spk.train.ReduceLROnPlateauHook(patience=args.lr_patience, factor=args.lr_decay,
                                               min_lr=args.lr_min,
                                               window_length=1, stop_after_min=True)
    hooks.append(schedule)

    # index into model output: [energy, forces]

    metrics = [spk.metrics.MeanAbsoluteError('energy', 'energy'),
               spk.metrics.RootMeanSquaredError('energy','energy'),
               spk.metrics.MeanAbsoluteError('forces', 'forces'),
               spk.metrics.RootMeanSquaredError('forces', 'forces'),
               spk.metrics.MeanAbsoluteError('dipoles', 'dipoles'),
               spk.metrics.RootMeanSquaredError('dipoles', 'dipoles'),
               spk.metrics.MeanAbsoluteError('nacs','nacs'),
               spk.metrics.RootMeanSquaredError('nacs','nacs')]
    if args.logger == 'csv':
        logger = spk.train.CSVHook(os.path.join(args.modelpath, 'log'),
                                   metrics, every_n_epochs=args.log_every_n_epochs)
        hooks.append(logger)
    elif args.logger == 'tensorboard':
        logger = spk.train.TensorboardHook(os.path.join(args.modelpath, 'log'),
                                           metrics, every_n_epochs=args.log_every_n_epochs)
        hooks.append(logger)

    #setup loss function
    def loss(batch, result):
        ediff = batch['energy'] - result['energy']
        ediff = ediff ** 2
        fdiff = batch['forces'] - result['forces']
        fdiff = fdiff ** 2

        ddiff = batch['dipoles'] - result['dipoles']
        ddiff = ddiff ** 2
        ndiff = batch['nacs'] - result['nacs']
        ndiff = ndiff ** 2
        err_sq = args.rho * torch.mean(ediff.view(-1)) + (1 - args.rho) * \
<<<<<<< HEAD
                 torch.mean(fdiff.view(-1)) + args.rho_dipole * torch.mean(ddiff.view(-1)) \
                 + args.rho_nac * torch.mean(ndiff.view(-1))
=======
                 torch.mean(fdiff.view(-1))

        t_diff = torch.mean(result['tloss'].view(-1))
        err_sq = err_sq + 0.01 * t_diff

        d_diff = torch.mean(result['off_diag'].view(-1))
        print(d_diff, 'D')
        err_sq = err_sq + 0.01 * d_diff
>>>>>>> 38e4f778d9b3a7ee5ee8af081ff16cd09b98dd05

        return err_sq

    trainer = spk.train.Trainer(args.modelpath, model, loss, optimizer,
                                train_loader, val_loader, hooks=hooks)
    trainer.train(device)


<<<<<<< HEAD
def QMout(prediction,modelpath):
    #returns predictions in QM.out format useable with SHARC
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


def run_prediction(modle,loader,device,args):
=======
def run_prediction(model, loader, device, args):
>>>>>>> 38e4f778d9b3a7ee5ee8af081ff16cd09b98dd05
    from tqdm import tqdm
    import numpy as np

    predicted = {}

    for batch in tqdm(loader, ncols=120):
        batch = {
            k: v.to(device)
<<<<<<< HEAD
            for k,v in batch.items()
        }
        result   = model(batch)
        energies = result['energy'].cpu().detach().numpy()
        forces   = result['forces'].cpu().detach().numpy()
        dipoles  = result['dipoles'].cpu().detach().numpy()
        nacs     = result['nacs'].cpu().detach().numpy()

=======
            for k, v in batch.items()
        }
        result = model(batch)
        print(result)

        energies = result['y'].cpu().detach().numpy()
        forces = result['dydx'].cpu().detach().numpy()
>>>>>>> 38e4f778d9b3a7ee5ee8af081ff16cd09b98dd05

        if 'energy' in predicted.keys():
            predicted['energy'].append(energies)
        else:
            predicted['energy'] = [energies]

        if 'force' in predicted.keys():
            predicted['force'].append(forces)
        else:
            predicted['force'] = [forces]

<<<<<<< HEAD
        if 'dipoles' in predicted.keys():
            predicted['dipoles'].append(dipoles)
        else:
            predicted['dipoles'] = [dipoles]

        if 'nacs' in predicted.keys():
            predicted['nacs'].append(nacs)
        else:
            predicted['nacs'] = [nacs]
    for p in predicted.keys():
        predicted[p] = np.vstack(predicted[p])
    np.savez(os.path.join(args.modelpath, 'predictions.npz'), **predicted)
    return predicted

=======
        if args.model == 'sort':
            all_e = result['y_all'].cpu().detach().numpy()
            if 'all_energies' in predicted.keys():
                predicted['all_energies'] += [all_e]
            else:
                predicted['all_energies'] = [all_e]

            all_me = result['y_all_mixed'].cpu().detach().numpy()
            if 'all_energies_mixed' in predicted.keys():
                predicted['all_energies_mixed'] += [all_me]
            else:
                predicted['all_energies_mixed'] = [all_me]

            coeff = result['c_states'].cpu().detach().numpy()
            if 'state' in predicted.keys():
                predicted['state'] += [coeff]
            else:
                predicted['state'] = [coeff]

    for p in predicted.keys():
        predicted[p] = np.vstack(predicted[p])

    np.savez(os.path.join(args.modelpath, 'predictions.npz'), **predicted)
    print('HIHI')
>>>>>>> 38e4f778d9b3a7ee5ee8af081ff16cd09b98dd05


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
                        help='Energy-force trade-off. For rho=0, use forces only. (default: %(default)s)',
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
    choices = ['energy', 'force', 'dipoles', 'nacs']
    parser.add_argument('--properties', type=str, help="Possible properties: energy, force, dipoles, nacs", default=choices)
    parser.add_argument('--overwrite', action='store_true', help='Overwrite old directories')
    parser.add_argument('--n_features', default=256, type=int, help='Number of features used by SchNet.')
    parser.add_argument('--n_interactions', default=6, type=int, help='Number of interactions used by SchNet.')
    parser.add_argument('--return_QMout', action='store_true', help='print result in QM.out format of SHARC')
    parser.add_argument('--schnarc', action='store_true',help='run schnarc')
    args = parser.parse_args()

    # Get device
    device = torch.device("cuda" if args.cuda else "cpu")

    # Load data
    data = spk.data.AtomsData(args.database, required_properties=['energy',
                                                                  'forces',
                                                                  'dipoles',
                                                                  'nacs'])
    if args.schnarc:
        data = spk.data.AtomsData(args.database, required_properties=[])
        loader = spk.data.AtomsLoader(data,batch_size=args.batch_size,num_workers=2,pin_memory=True)
        model = torch.load(os.path.join(args.modelpath, 'best_model')).to(device)
        prediction = run_prediction(model,loader,device,args)
        QMout = get_schnarc(prediction,args.properties)
        exit()
    if args.evaluate:
        data = spk.data.AtomsData(args.database, required_properties=[])
        loader = spk.data.AtomsLoader(data,batch_size=args.batch_size,num_workers=2,pin_memory=True)
        model = torch.load(os.path.join(args.modelpath, 'best_model')).to(device)
        prediction=run_prediction(model,loader,device,args)
        if args.return_QMout:
            os.system("mkdir %s/prediction" %args.modelpath)
            QMout(prediction,args.modelpath+"/prediction")
        exit()

    if args.evaluate:
        data = spk.data.AtomsData(args.database, required_properties=[])
        loader = spk.data.AtomsLoader(data, batch_size=args.batch_size, num_workers=2, pin_memory=True)
        model = torch.load(os.path.join(args.modelpath, 'best_model')).to(device)
        run_prediction(model, loader, device, args)
        exit()

    if args.overwrite:
        if os.path.exists(args.modelpath):
            shutil.rmtree(args.modelpath)


    n_states = data[0]['energy'].shape[0]
    #n_atoms  = data[0]['nacs'].shape[1]
    n_dipole = int((n_states*(n_states-1)/2+n_states)) # nstates*(nstates-1)/2 +nstates only upper triangular matrix + diagonal
    #n_nacs   = int(n_atoms*(n_states*(n_states-1)/2)) # nstates*(nstates-1)/2 different matrices of natoms x 3 
    # Create splits
    train_data, valid_data, test_data = data.create_splits(*args.split)

    # Create loaders
    train_loader = spk.data.AtomsLoader(
        train_data, batch_size=args.batch_size, sampler=RandomSampler(train_data),
        num_workers=4, pin_memory=True)
    val_loader = spk.data.AtomsLoader(valid_data, batch_size=args.batch_size, num_workers=2, pin_memory=True)

<<<<<<< HEAD
    # Get statistics - not of vectors like gradients, dipoles or NACs
    mean, stddev = train_loader.get_statistics('energy', True)
    print(mean, stddev)

    representation = spk.representation.SchNet(args.n_features, args.n_features, args.n_interactions, 15.0, 25)
=======
    # Get statistics
    mean, stddev = train_loader.get_statistics('energy', per_atom=False)
    print(mean, stddev)

    representation = spk.representation.SchNet(args.n_features, args.n_features, args.n_interactions, 18.0, 50,
                                               cutoff_network=spk.nn.CosineCutoff)

    # New convention
    # mod = {
    #     'energy': HiddenStatesEnergy(args.n_features, n_states, mean=mean, stddev=stddev, return_force=True, create_graph=True)
    # }
    # map = {'forces': ('energy', 'dydx')}

    # model = StateModel(representation, mod, mapping=map).to(device)

    if args.model == 'standard':
        atomwise_output = MultiEnergy(args.n_features, n_states, mean=mean, stddev=stddev, return_force=True,
                                      create_graph=True)
    elif args.model == 'sort':
        atomwise_output = HiddenStatesEnergy(args.n_features, n_states, mean=mean, stddev=stddev, return_force=True,
                                             create_graph=True)
    else:
        raise NotImplementedError
>>>>>>> 38e4f778d9b3a7ee5ee8af081ff16cd09b98dd05

    #atomwise_output = [MultiEnergy(args.n_features, n_states, mean=mean, stddev=stddev, return_force=True,
    #                              create_graph=True),MultiNac()]
    prop_dict ={
    'energy' : MultiEnergy(args.n_features,n_states,mean=mean,stddev=stddev,return_force=True,create_graph=True),
    'dipoles': MultiDipole(args.n_features,n_dipole,mean=None,stddev=None,return_force=False),
    'nacs'   : MultiNac(args.n_features,n_states,mean=None,stddev=None,return_force=True,create_graph=True)}
    atomwise = [prop_dict[p] for p in prop_dict.keys()]
    #atomwise_output=[MultiEnergy(args.n_features, n_states, mean=mean, stddev=stddev, return_force=True, create_graph=True),
    #                  MultiDipole(args.n_features, n_dipole, mean=None, stddev=None, return_force=False,create_graph=True)]
    model = CombineProperties(representation,atomwise,prop_dict).to(device)
    #model = spk.atomistic.AtomisticModel(representation, atomwise_output).to(device)
    #model = spk.atomistic.AtomisticModel(representation, atomwise).to(device)

    train(args, model, train_loader, val_loader, device,prop_dict)
