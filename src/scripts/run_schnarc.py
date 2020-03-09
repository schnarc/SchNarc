#!/usr/bin/env python
import argparse
import logging
import os
import sys
import numpy as np
from shutil import copyfile, rmtree
from ase import units
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data.sampler import RandomSampler

import schnetpack as spk
import schnarc

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def get_parser():
    """ Setup parser for command line arguments """
    main_parser = argparse.ArgumentParser()

    # command-specific
    cmd_parser = argparse.ArgumentParser(add_help=False)
    cmd_parser.add_argument('--cuda', help='Set flag to use GPU(s)', action='store_true')
    cmd_parser.add_argument('--parallel',
                            help='Run data-parallel on all available GPUs (specify with environment variable'
                                 + ' CUDA_VISIBLE_DEVICES)', action='store_true')
    cmd_parser.add_argument('--batch_size', type=int,
                            help='Mini-batch size for training and prediction (default: %(default)s)',
                            default=100)

    # training
    train_parser = argparse.ArgumentParser(add_help=False, parents=[cmd_parser])
    train_parser.add_argument('datapath', help='Path / destination of MD17 dataset')
    train_parser.add_argument('modelpath', help='Destination for models and logs')
    train_parser.add_argument('--seed', type=int, default=None, help='Set random seed for torch and numpy.')
    train_parser.add_argument('--overwrite', action='store_true', help='Overwrite old directories')

    # data split
    train_parser.add_argument('--split_path', help='Path / destination of npz with data splits',
                              default=None)
    train_parser.add_argument('--split', help='Give sizes of train and validation splits and use remaining for testing',
                              type=int, nargs=2, default=[None, None])
    train_parser.add_argument('--max_epochs', type=int, help='Maximum number of training epochs (default: %(default)s)',
                              default=5000)
    train_parser.add_argument('--lr', type=float, help='Initial learning rate (default: %(default)s)',
                              default=1e-4)
    train_parser.add_argument('--lr_patience', type=int,
                              help='Epochs without improvement before reducing the learning rate (default: %(default)s)',
                              default=15)
    train_parser.add_argument('--lr_decay', type=float, help='Learning rate decay (default: %(default)s)',
                              default=0.8)
    train_parser.add_argument('--lr_min', type=float, help='Minimal learning rate (default: %(default)s)',
                              default=1e-6)
    train_parser.add_argument('--logger', help='Choose logger for training process (default: %(default)s)',
                              choices=['csv', 'tensorboard'], default='csv')
    train_parser.add_argument('--log_every_n_epochs', type=int,
                              help='Log metrics every given number of epochs (default: %(default)s)',
                              default=1)
    train_parser.add_argument('--tradeoffs', type=str, help='Property tradeoffs for training', default=None)
    train_parser.add_argument('--verbose', action='store_true', help='Print property error magnitudes')
    train_parser.add_argument('--real_socs', action='store_true',
                              help='If spin-orbit couplings are predicted, information should be given whether they are real or not.')
    train_parser.add_argument('--phase_loss', action='store_true', help='Use special loss, which ignores phase.')
    train_parser.add_argument('--inverse_nacs', action='store_true', help='Weight NACs with inverse energies.')
    train_parser.add_argument('--min_loss', action='store_true', help='Use phase independent min loss.')
    train_parser.add_argument('--L1', action='store_true', help='Use L1 norm')
    train_parser.add_argument('--Huber', action='store_true', help='Use L1 norm')

    ## evaluation
    eval_parser = argparse.ArgumentParser(add_help=False, parents=[cmd_parser])
    eval_parser.add_argument('datapath', help='Path / destination of MD17 dataset directory')
    eval_parser.add_argument('modelpath', help='Path of stored model')
    eval_parser.add_argument('--split', help='Evaluate on trained model on given split',
                             choices=['train', 'validation', 'test', 'all'], default=['all'], nargs='+')
    eval_parser.add_argument('--hessian', action='store_true', help='Gives back the hessian of the PES.')

    pred_parser = argparse.ArgumentParser(add_help=False, parents=[cmd_parser])
    pred_parser.add_argument('datapath', help='Path / destination of MD17 dataset directory')
    pred_parser.add_argument('modelpath', help='Path of stored model')
    pred_parser.add_argument('--hessian', action='store_true', help='Gives back the hessian of the PES.')
    pred_parser.add_argument('--nac_approx',type=float, nargs=3, default=[1,0.018,0.036],help='Type of NAC approximation as first value and threshold for energy gap in Hartree as second value.')
    # model-specific parsers
    model_parser = argparse.ArgumentParser(add_help=False)

    #######  SchNet  #######
    schnet_parser = argparse.ArgumentParser(add_help=False, parents=[model_parser])
    schnet_parser.add_argument('--features', type=int, help='Size of atom-wise representation (default: %(default)s)',
                               default=256)
    schnet_parser.add_argument('--interactions', type=int, help='Number of interaction blocks (default: %(default)s)',
                               default=6)
    schnet_parser.add_argument('--cutoff', type=float, default=10.,
                               help='Cutoff radius of local environment (default: %(default)s)')
    schnet_parser.add_argument('--num_gaussians', type=int, default=50,
                               help='Number of Gaussians to expand distances (default: %(default)s)')
    schnet_parser.add_argument('--n_layers', type=int, default=3,
                               help='Number of layers in output networks (default: %(default)s)')

    #######  wACSF  ########
    wacsf_parser = argparse.ArgumentParser(add_help=False, parents=[model_parser])
    # wACSF parameters
    wacsf_parser.add_argument('--radial', type=int, default=22,
                              help='Number of radial symmetry functions (default: %(default)s)')
    wacsf_parser.add_argument('--angular', type=int, default=5,
                              help='Number of angular symmetry functions (default: %(default)s)')
    wacsf_parser.add_argument('--zetas', type=int, nargs='+', default=[1],
                              help='List of zeta exponents used for angle resolution (default: %(default)s)')
    wacsf_parser.add_argument('--standardize', action='store_true',
                              help='Standardize wACSF before atomistic network.')
    wacsf_parser.add_argument('--cutoff', type=float, default=5.0,
                              help='Cutoff radius of local environment (default: %(default)s)')
    # Atomistic network parameters
    wacsf_parser.add_argument('--n_nodes', type=int, default=100,
                              help='Number of nodes in atomic networks (default: %(default)s)')
    wacsf_parser.add_argument('--n_layers', type=int, default=2,
                              help='Number of layers in atomic networks (default: %(default)s)')
    # Advances wACSF settings
    wacsf_parser.add_argument('--centered', action='store_true', help='Use centered Gaussians for radial functions')
    wacsf_parser.add_argument('--crossterms', action='store_true', help='Use crossterms in angular functions')
    wacsf_parser.add_argument('--behler', action='store_true', help='Switch to conventional ACSF')
    wacsf_parser.add_argument('--elements', default=['H', 'C', 'O'], nargs='+',
                              help='List of elements to be used for symmetry functions (default: %(default)s).')

    # setup subparser structure
    cmd_subparsers = main_parser.add_subparsers(dest='mode', help='Command-specific arguments')
    cmd_subparsers.required = True
    subparser_train = cmd_subparsers.add_parser('train', help='Training help')
    subparser_eval = cmd_subparsers.add_parser('eval', help='Eval help')

    subparser_pred = cmd_subparsers.add_parser('pred', help='Eval help', parents=[pred_parser])

    train_subparsers = subparser_train.add_subparsers(dest='model', help='Model-specific arguments')
    train_subparsers.required = True
    train_subparsers.add_parser('schnet', help='SchNet help', parents=[train_parser, schnet_parser])
    train_subparsers.add_parser('wacsf', help='wACSF help', parents=[train_parser, wacsf_parser])

    eval_subparsers = subparser_eval.add_subparsers(dest='model', help='Model-specific arguments')
    eval_subparsers.required = True
    eval_subparsers.add_parser('schnet', help='SchNet help', parents=[eval_parser, schnet_parser])
    eval_subparsers.add_parser('wacsf', help='wACSF help', parents=[eval_parser, wacsf_parser])

    return main_parser


def train(args, model, tradeoffs, train_loader, val_loader, device, n_states, props_phase):
    # setup hook and logging
    hooks = [
        spk.train.MaxEpochHook(args.max_epochs)
    ]

    # setup optimizer for training
    to_opt = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(to_opt, lr=args.lr)

    schedule = spk.train.ReduceLROnPlateauHook(optimizer, patience=args.lr_patience, factor=args.lr_decay,
                                               min_lr=args.lr_min,
                                               window_length=1, stop_after_min=True)
    hooks.append(schedule)

    # Build metrics based on the properties in the model
    metrics = []
    for prop in tradeoffs:
       if prop=="socs":
           socs_given=True
       else:
           socs_given=False
    for prop in tradeoffs:
       if args.phase_loss or args.min_loss:
            if prop in schnarc.data.Properties.phase_properties:
                if prop == 'nacs' and socs_given == True:
                    metrics += [
                        schnarc.metrics.PhaseMeanAbsoluteError(prop, prop),
                        schnarc.metrics.PhaseRootMeanSquaredError(prop, prop)
                    ]
                else:
                    metrics += [
                        schnarc.metrics.PhaseMeanAbsoluteError(prop, prop),
                        schnarc.metrics.PhaseRootMeanSquaredError(prop, prop)
                    ]
            else:
                metrics += [
                    spk.metrics.MeanAbsoluteError(prop, prop),
                    spk.metrics.RootMeanSquaredError(prop, prop)
                ]
       else:
          metrics += [
              spk.metrics.MeanAbsoluteError(prop, prop),
              spk.metrics.RootMeanSquaredError(prop, prop)
          ]
    if args.logger == 'csv':
        logger = spk.train.CSVHook(os.path.join(args.modelpath, 'log'),
                                   metrics, every_n_epochs=args.log_every_n_epochs)
        hooks.append(logger)
    elif args.logger == 'tensorboard':
        logger = spk.train.TensorboardHook(os.path.join(args.modelpath, 'log'),
                                           metrics, every_n_epochs=args.log_every_n_epochs)
        hooks.append(logger)

    # Automatically construct loss function based on tradoffs
    def loss(batch, result):
        err_sq = 0.0
        if args.verbose:
            print('===================')
        if "socs" in tradeoffs and "nacs" in tradeoffs:
            combined_phaseless_loss = True
        else:
            combined_phaseless_loss = False
        for prop in tradeoffs:
            if args.min_loss and prop in schnarc.data.Properties.phase_properties:
                if prop == "socs" and combined_phaseless_loss == True:
                    prop_diff = schnarc.nn.min_loss(batch[prop], result[prop], combined_phaseless_loss, n_states, props_phase, phase_vector_nacs )
                    #already spared and mean of all values
                    prop_err = torch.mean(prop_diff.view(-1))
                elif prop == "socs" and combined_phaseless_loss == False:
                    #prop_err = schnarc.nn.min_loss_single_old(batch[prop], result[prop], smooth=False, smooth_nonvec=False, loss_length=False)
                    prop_diff = schnarc.nn.min_loss_single(batch[prop], result[prop], combined_phaseless_loss, n_states, props_phase )
                    prop_err = torch.mean(prop_diff.view(-1) **2 )
                elif prop == "dipoles" and combined_phaseless_loss == True:
                    prop_err = schnarc.nn.min_loss(batch[prop], result[prop], combined_phaseless_loss, n_states, props_phase, phase_vector_nacs, dipole=True )
                    #already spared and mean of all values
                    prop_err = torch.mean(prop_diff.view(-1))
                elif prop == "dipoles" and combined_phaseless_loss == False:
                    #prop_err = schnarc.nn.min_loss_single_old(batch[prop], result[prop],loss_length=False)
                    prop_diff = schnarc.nn.min_loss_single(batch[prop], result[prop], combined_phaseless_loss, n_states, props_phase, dipole = True )
                    prop_err = torch.mean(prop_diff.view(-1) **2 )
                elif prop == "nacs" and combined_phaseless_loss == True:
                    #for nacs regardless of any other available phase-property
                    prop_diff, phase_vector_nacs = schnarc.nn.min_loss(batch[prop], result[prop], False, n_states, props_phase)
                    prop_err = torch.mean(prop_diff.view(-1)) / (2*n_states['n_states']**2)
                else:
                    prop_diff, phase_vector_nacs = schnarc.nn.min_loss(batch[prop], result[prop], False, n_states, props_phase)
                    prop_err = torch.mean(prop_diff.view(-1) **2 )
                    #prop_err = schnarc.nn.min_loss_single(batch[prop], result[prop],loss_length=False)

            elif args.L1 and prop == schnarc.data.Properties.energy or args.L1 and prop == schnarc.data.Properties.forces:
                prop_diff = torch.abs(batch[prop] - result[prop])
                prop_err = torch.mean(prop_diff.view(-1) )
            elif args.Huber and prop == schnarc.data.Properties.energy or args.Huber and prop == schnarc.data.Properties.forces:
                prop_diff = torch.abs(batch[prop]-result[prop])
                if torch.mean(prop_diff.view(-1)) <= 0.005 and prop == schnarc.data.Properties.forces:
                    prop_err = torch.mean(prop_diff.view(-1) **2 )
                elif torch.mean(prop_diff.view(-1)) <= 0.05 and prop == schnarc.data.Properties.energy:
                    prop_err = torch.mean(prop_diff.view(-1) **2 )
                else:
                    prop_err = torch.mean(prop_diff.view(-1))
            else:
                if prop=='energy' and result['forces'].shape[1]==int(1) or prop=='forces' and result['forces'].shape[1]==int(1) or prop=='dipoles' and result['forces'].shape[1]==int(1):
                    prop_diff = batch[prop][:,0] - result[prop][:,0]
                else:
                    prop_diff = batch[prop] - result[prop]
                prop_err = torch.mean(prop_diff.view(-1) ** 2)
            err_sq = err_sq + float(tradeoffs[prop].split()[0]) * prop_err
            if args.verbose:
                print(prop, prop_err)

        return err_sq

    trainer = spk.train.Trainer(args.modelpath, model, loss, optimizer,
                                train_loader, val_loader, hooks=hooks)
    trainer.train(device)


def evaluate(args, model, train_loader, val_loader, test_loader, device):
    # Get property names from model
    if args.parallel:
        properties=model.module.output_modules.properties
    else:
        properties = model.output_modules.properties
    header = ['Subset']
    metrics = []
    for prop in properties:
      if prop=="dipoles":
        pass
      else:
        header += [f'{prop}_MAE', f'{prop}_RMSE']
        if prop in schnarc.data.Properties.phase_properties:
            header += [f'{prop}_pMAE', f'{prop}_pRMSE']
            metrics += [
                schnarc.metrics.PhaseMeanAbsoluteError(prop, prop),
                schnarc.metrics.PhaseRootMeanSquaredError(prop, prop)
            ]
        else:
            metrics += [
                schnarc.metrics.MeanAbsoluteError(prop, prop),
                schnarc.metrics.RootMeanSquaredError(prop, prop)
            ]

    results = []
    if ('train' in args.split) or ('all' in args.split):
        logging.info('Training split...')
        results.append(['training'] + ['%.7f' % i for i in evaluate_dataset(metrics, model, train_loader, device,properties)])

    if ('validation' in args.split) or ('all' in args.split):
        logging.info('Validation split...')
        results.append(['validation'] + ['%.7f' % i for i in evaluate_dataset(metrics, model, val_loader, device,properties)])

    if ('test' in args.split) or ('all' in args.split):
        logging.info('Testing split...')
        results.append(['test'] + ['%.7f' % i for i in evaluate_dataset(metrics, model, test_loader, device,properties)])

    header = ','.join(header)
    results = np.array(results)

    np.savetxt(os.path.join(args.modelpath, 'evaluation.csv'), results, header=header, fmt='%s', delimiter=',')

def evaluate_dataset(metrics, model, loader, device,properties):
    # TODO: Adapt for SCHNARC, copy old
    for metric in metrics:
        metric.reset()

    predicted={}
    header=[]
    for batch in loader:
        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }
        result = model(batch)
        for metric in metrics:
            metric.add_batch(batch, result)
    results = [
    metric.aggregate() for metric in metrics
    ]


    return results


def run_prediction(model, loader, device, args):
    from tqdm import tqdm
    import numpy as np

    predicted = {}

    for batch in tqdm(loader, ncols=120):
        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }
        result = model(batch)
        for prop in result:
            if prop in predicted:
                predicted[prop] += [result[prop].cpu().detach().numpy()]
            else:
                predicted[prop] = [result[prop].cpu().detach().numpy()]

    for p in predicted.keys():
        predicted[p] = np.vstack(predicted[p])

    prediction_path = os.path.join(args.modelpath, 'predictions.npz')
    np.savez(prediction_path, **predicted)
    logging.info('Stored model predictions in {:s}...'.format(prediction_path))


def get_model(args, n_states, properties, atomref=None, mean=None, stddev=None, train_loader=None, parallelize=False,
              mode='train'):
    if args.model == 'schnet':
        representation = spk.representation.SchNet(args.features, args.features, args.interactions,
                                                   args.cutoff / units.Bohr, args.num_gaussians)

        property_output = schnarc.model.MultiStatePropertyModel(args.features, n_states, properties=properties,
                                                                mean=mean, stddev=stddev, atomref=atomref,
                                                                n_layers=args.n_layers, real=args.real_socs,
                                                                inverse_energy=args.inverse_nacs)

        model = spk.atomistic.AtomisticModel(representation, property_output)

    elif args.model == 'wacsf':
        raise NotImplementedError
        # from ase.data import atomic_numbers
        # sfmode = ('weighted', 'Behler')[args.behler]
        # # Convert element strings to atomic charges
        # elements = frozenset((atomic_numbers[i] for i in sorted(args.elements)))
        # representation = spk.representation.BehlerSFBlock(args.radial, args.angular, zetas=set(args.zetas),
        #                                                   cutoff_radius=args.cutoff,
        #                                                   centered=args.centered, crossterms=args.crossterms,
        #                                                   elements=elements,
        #                                                   mode=sfmode)
        # logging.info("Using {:d} {:s}-type SF".format(representation.n_symfuncs, sfmode))
        # # Standardize representation if requested
        # if args.standardize and mode == 'train':
        #     if train_loader is None:
        #         raise ValueError("Specification of a trainig_loader is required to standardize wACSF")
        #     else:
        #         logging.info("Computing and standardizing symmetry function statistics")
        # else:
        #     train_loader = None
        #
        # representation = spk.representation.StandardizeSF(representation, train_loader, cuda=args.cuda)
        #
        # Build HDNN model

    # TODO: rework, can not use Behler SFs?, tedious for elemental networks...
    #     atomwise_output = spk.atomistic.ElementalEnergy(representation.n_symfuncs, n_hidden=args.n_nodes,
    #                                                     n_layers=args.n_layers, mean=mean, stddev=stddev,
    #                                                     atomref=atomref, return_force=True, create_graph=True,
    #                                                     elements=elements)
    #     model = spk.atomistic.AtomisticModel(representation, atomwise_output)

    else:
        raise ValueError('Unknown model class:', args.model)

    if parallelize:
        model = nn.DataParallel(model)

    logging.info("The model you built has: %d parameters" % spk.utils.count_params(model))

    return model

def printTheArray(arr, n, phasevector):
     phasevector=phasevector+arr
     return phasevector

def  generateAllBinaryStrings(n, arr, i, phasevector):

     if i == n:
        phasevector=printTheArray(arr, n, phasevector)
        return phasevector

     # First assign "0" at ith position
     # and try for all other permutations
     # for remaining positions
     arr[i] = 0
     phasevector=generateAllBinaryStrings(n, arr, i + 1, phasevector)

     # And then assign "1" at ith position
     # and try for all other permutations
     # for remaining positions
     arr[i] = 1
     phasevector=generateAllBinaryStrings(n, arr, i + 1, phasevector)
     return phasevector

def generateAllPhaseMatrices(phase_pytorch,n_states,n_socs,all_states,device):
    #generate a matrix that is the outer product of all possible phase combinations. hence this matrix can be directly multiplied with the hamiltonian matrix or dipole matrix
    #size is singlets+3*triplets
    phase_vector_nacs_1 = torch.Tensor(all_states,phase_pytorch.shape[1]).to(device)
    phase_vector_nacs_2 = torch.Tensor(all_states,phase_pytorch.shape[1]).to(device)
    phase_vector_nacs_1[:n_states['n_singlets']+n_states['n_triplets'],:] = phase_pytorch
    #append the triplet part
    phase_vector_nacs_1[n_states['n_singlets']+n_states['n_triplets']:n_states['n_singlets']+n_states['n_triplets']*2,:] = phase_pytorch[n_states['n_singlets']:,:]
    phase_vector_nacs_1[n_states['n_singlets']+n_states['n_triplets']*2:n_states['n_singlets']+n_states['n_triplets']*3,:] = phase_pytorch[n_states['n_singlets']:,:]

    phase_vector_nacs_2[:n_states['n_singlets'],:] = phase_vector_nacs_1[:n_states['n_singlets'],:]
    phase_vector_nacs_2[n_states['n_singlets']:,:] = phase_vector_nacs_1[n_states['n_singlets']:,:] * -1
    #two possibilities - the min function should be give the correct error
    #therefore, build a matrix that contains all the two possibilities of phases by building the outer product of each phase vector for     a given sample of a mini batch
    complex_diagonal_phase_matrix_1 = torch.Tensor(phase_pytorch.shape[1],n_socs).to(device)
    phase_matrix_1 = torch.Tensor(phase_pytorch.shape[1],all_states,all_states).to(device)
    complex_diagonal_phase_matrix_2 = torch.Tensor(phase_pytorch.shape[1],n_socs).to(device)
    phase_matrix_2 = torch.Tensor(phase_pytorch.shape[1],all_states,all_states).to(device)
    #build the phase matrix
    for possible_permutation in range(0,phase_pytorch.shape[1]):
        phase_matrix_1[possible_permutation,:,:] = torch.ger(phase_vector_nacs_1[:,possible_permutation],phase_vector_nacs_1[:,possible_permutation])
        phase_matrix_2[possible_permutation,:,:] = torch.ger(phase_vector_nacs_2[:,possible_permutation],phase_vector_nacs_2[:,possible_permutation])
    diagonal_phase_matrix_1=phase_matrix_1[:,torch.triu(torch.ones(all_states,all_states)) == 0]
    diagonal_phase_matrix_2=phase_matrix_2[:,torch.triu(torch.ones(all_states,all_states)) == 0]
    for i in range(int(n_socs/2)):
        complex_diagonal_phase_matrix_1[:,2*i] = diagonal_phase_matrix_1[:,i]
        complex_diagonal_phase_matrix_1[:,2*i+1] = diagonal_phase_matrix_1[:,i]
        complex_diagonal_phase_matrix_2[:,2*i] = diagonal_phase_matrix_2[:,i]
        complex_diagonal_phase_matrix_2[:,2*i+1] = diagonal_phase_matrix_2[:,i]
    return complex_diagonal_phase_matrix_1, complex_diagonal_phase_matrix_2, phase_matrix_1[:,torch.triu(torch.ones(all_states,all_states)) == 1], phase_matrix_2[:,torch.triu(torch.ones(all_states,all_states)) == 1]


if __name__ == '__main__':
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Determine the device
    device = torch.device("cuda" if args.cuda else "cpu")
    if args.mode != 'train':
        model = torch.load(os.path.join(args.modelpath, 'best_model'), map_location='cpu').to(device)
        if args.hessian == True:
            model.output_modules.output_dict['energy'].return_hessian = [True,1,1,1,1]
        else:
            model.output_modules.output_dict['energy'].return_hessian = [False,1,1]
 
    if args.mode == 'pred':
        pred_data = spk.data.AtomsData(args.datapath, required_properties=[])
        pred_loader = spk.data.AtomsLoader(pred_data, batch_size=args.batch_size, num_workers=2, pin_memory=True)
        run_prediction(model, pred_loader, device, args)
        sys.exit(0)

    # Load settings
    argparse_dict = vars(args)
    jsonpath = os.path.join(args.modelpath, 'args.json')

    # Setup directories and files for training or load settings for restart.
    if args.mode == 'train':
        if args.overwrite and os.path.exists(args.modelpath):
            logging.info('Existing model will be overwritten...')
            rmtree(args.modelpath)

        if not os.path.exists(args.modelpath):
            os.makedirs(args.modelpath)

        spk.utils.to_json(jsonpath, argparse_dict)
        spk.utils.set_random_seed(args.seed)
        train_args = args

        # Read the tradeoff file or generate a default file
        tradeoff_file = os.path.join(args.modelpath, 'tradeoffs.yaml')

        if train_args.tradeoffs is None:
            schnarc.utils.generate_default_tradeoffs(tradeoff_file)
            tradeoffs = schnarc.utils.read_tradeoffs(tradeoff_file)
        else:
            tradeoffs = schnarc.utils.read_tradeoffs(args.tradeoffs)
            schnarc.utils.save_tradeoffs(tradeoffs, tradeoff_file)
    else:
        train_args = spk.utils.read_from_json(jsonpath)
        tradeoff_file = os.path.join(args.modelpath, 'tradeoffs.yaml')
        tradeoffs = schnarc.utils.read_tradeoffs(tradeoff_file)

    # Determine the properties to load based on the tradeoffs
    properties = [p for p in tradeoffs]

    # Read and process the data using the properties found in the tradeoffs.
    logging.info('Loading {:s}...'.format(args.datapath))
    dataset = spk.data.AtomsData(args.datapath, collect_triples=args.model == 'wacsf')
    # Determine the number of states based on the metadata
    n_states = {}
    n_states['n_singlets'] = dataset.get_metadata("n_singlets")
    if dataset.get_metadata("n_triplets") == None:
        n_states['n_triplets']=int(0)
    else:
        n_states['n_triplets'] = dataset.get_metadata("n_triplets")
    n_states['n_states'] = n_states['n_singlets'] + n_states['n_triplets']

    ##activate if only one state is learned or not all
    s=tradeoffs['energy'].split()
    if int(s[1]) > int(0):
        n_singlets = int(s[1])
        n_triplets = int(s[2])
        n_states['n_singlets'] = n_singlets
        n_states['n_triplets'] = n_triplets
        n_states['n_states'] = n_states['n_singlets'] + n_states['n_triplets']
    n_states['states'] = dataset.get_metadata("states")
    logging.info('Found {:d} states... {:d} singlet states and {:d} triplet states'.format(n_states['n_states'],
                                                                                           n_states['n_singlets'],
                                                                                           n_states['n_triplets']))

    if args.mode == 'eval':
        split_path = os.path.join(args.modelpath, 'split.npz')
        data_train, data_val, data_test = dataset.create_splits(*args.split, split_file=split_path)

        train_loader = spk.data.AtomsLoader(data_train, batch_size=args.batch_size, sampler=RandomSampler(data_train),
                                        num_workers=4, pin_memory=True)
        val_loader = spk.data.AtomsLoader(data_val, batch_size=args.batch_size, num_workers=2, pin_memory=True)

        logging.info("evaluating...")
        test_loader = spk.data.AtomsLoader(data_test, batch_size=args.batch_size,
                                           num_workers=2, pin_memory=True)
        evaluate(args, model, train_loader, val_loader, test_loader, device)
        logging.info("... done!")
        exit()
     # Splits the dataset in test, val, train sets
    split_path = os.path.join(args.modelpath, 'split.npz')
 
    if args.mode == 'train':
        if args.split_path is not None:
            copyfile(args.split_path, split_path)

    logging.info('Creating splits...')
    data_train, data_val, data_test = dataset.create_splits(*train_args.split, split_file=split_path)

    # Generate loaders for training
    train_loader = spk.data.AtomsLoader(data_train, batch_size=args.batch_size, sampler=RandomSampler(data_train),
                                        num_workers=4, pin_memory=True)
    val_loader = spk.data.AtomsLoader(data_val, batch_size=args.batch_size, num_workers=2, pin_memory=True)

    # Determine statistics for scaling
    if args.mode == 'train':
        # Set the random seed
        spk.utils.set_random_seed(args.seed)

        logging.info('Calculate statistics...')
        mean = {}
        stddev = {}
        # Compute mean and stddev for relevant properties
        for p in properties:
            if p in schnarc.data.Properties.normalize:
                mean_p, stddev_p = train_loader.get_statistics(p, True)
                mean_p = mean_p[p]
                stddev_p = stddev_p[p]
                mean[p] = mean_p
                stddev[p] = stddev_p
                logging.info('{:s} MEAN: {:20.11f} STDDEV: {:20.11f}'.format(p, mean_p.detach().cpu().numpy()[0],
                                                                             stddev_p.detach().cpu().numpy()[0]))
    else:
        mean, stddev = None, None

    # Construct the model.
    model = get_model(train_args,
                      n_states,
                      properties,
                      mean=mean,
                      stddev=stddev,
                      train_loader=train_loader,
                      parallelize=args.parallel,
                      mode=args.mode
                      ).to(device)

    if args.mode == 'train':
        #model=torch.load(os.path.join(args.modelpath,'best_model'),map_location='cpu').to(device)
        logging.info("training...")
        #properties for phase vector
        #n_nacs = int(n_states['n_singlets']*(n_states['n_singlets']-1)/2 + n_states['n_triplets']*(n_states['n_triplets']-1)/2 )
        batch_size = args.batch_size
        all_states = n_states['n_singlets'] + 3 * n_states['n_triplets']
        n_socs = int(all_states * (all_states - 1)) # complex so no division by zero
        #min loss for a given batch size
        #vector with correct phases for a mini batch
        #number of possible phase vectors
        n_phases = int(2**(n_states['n_states']-1))
        #number of singlet-singlet and triplet-triplet deriative couplings
        #gives the number of phases 
        #generate all possible 2**(nstates-1) vectors that give rise to possible combinations of phases

        phasevector = generateAllBinaryStrings(n_states['n_states'],[None]*n_states['n_states'],0,[])
        phase_pytorch = torch.Tensor( n_states['n_states'],n_phases ).to(device)
        iterator = -1
        for i in range(n_phases):
            for j in range(n_states['n_states']):
                iterator+=1
                if phasevector[iterator]==0:
                    phase_pytorch[j,i] = 1
                else:
                    phase_pytorch[j,i] = -1

        socs_phase_matrix_1, socs_phase_matrix_2, diag_phase_matrix_1, diag_phase_matrix_2 = generateAllPhaseMatrices(phase_pytorch,n_states,n_socs,all_states,device)

        props_phase=[n_phases,batch_size,device,phase_pytorch,n_socs, all_states, socs_phase_matrix_1, socs_phase_matrix_2, diag_phase_matrix_1, diag_phase_matrix_2]
        train(args, model, tradeoffs, train_loader, val_loader, device, n_states,props_phase)
        logging.info("...training done!")

