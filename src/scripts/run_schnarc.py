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
from schnetpack.utils.script_utils.settings import get_environment_provider
from schnetpack.utils import get_loaders
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
    cmd_parser.add_argument("--environment_provider", type=str,default="simple",choices=["simple","ase","torch"],help="Environment provider for dataset (default: %(defaullt)s)",)
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
    train_parser.add_argument('--transfer', type=str, help='Previous training set used to compute mean', default=None)
    train_parser.add_argument('--tradeoffs', type=str, help='Property tradeoffs for training', default=None)
    train_parser.add_argument('--verbose', action='store_true', help='Print property error magnitudes')
    train_parser.add_argument('--real_socs', action='store_true',
                              help='If spin-orbit couplings are predicted, information should be given whether they are real or not.')
    train_parser.add_argument('--no_negative_dr', action='store_true', help='Train gradients instead of forces.')
    train_parser.add_argument('--phase_loss', action='store_true', help='Use special loss, which ignores phase.')
    train_parser.add_argument('--inverse_nacs', type=int, help='Weight NACs with inverse energies. 0 = False, 1 = first run (use QC energies), 2 = second run (use ML energies).', default = 0)
    train_parser.add_argument('--log', action='store_true', help='Use phase independent min loss.',default=False)
    train_parser.add_argument('--min_loss', action='store_true', help='Use phase independent min loss.')
    train_parser.add_argument('--diagonal', action='store_true', help='Train SOCs via diagonal elements. Must be included in the training data base', default=None)
    train_parser.add_argument('--L1', action='store_true', help='Use L1 norm')
    train_parser.add_argument('--Huber', action='store_true', help='Use L1 norm')
    train_parser.add_argument("--order",action="store_true",help="orders states by energy.")
    train_parser.add_argument("--finish",action="store_true",help="assume that the dynamics will only occur between S1 and S0.")

    ## evaluation
    eval_parser = argparse.ArgumentParser(add_help=False, parents=[cmd_parser])
    eval_parser.add_argument('datapath', help='Path / destination of MD17 dataset directory')
    eval_parser.add_argument("--order",action="store_true",help="orders states by energy.")
    eval_parser.add_argument('modelpath', help='Path of stored model')
    eval_parser.add_argument('--log', action='store_true', help='Use phase independent min loss.',default=False)
    eval_parser.add_argument('--split', help='Evaluate on trained model on given split',
                             choices=['train', 'validation', 'test', 'all'], default=['all'], nargs='+')
    eval_parser.add_argument('--hessian', action='store_true', help='Gives back the hessian of the PES.')
    eval_parser.add_argument("--finish",action="store_true",help="assume that the dynamics will only occur between S1 and S0.")

    pred_parser = argparse.ArgumentParser(add_help=False, parents=[cmd_parser])
    pred_parser.add_argument("--order",action="store_true",help="orders states by energy.")
    pred_parser.add_argument('datapath', help='Path / destination of MD17 dataset directory')
    pred_parser.add_argument('--print_uncertainty',action='store_true', help='Print uncertainty of each property',default=False)
    #pred_parser.add_argument('--thresholds',type=float, help='Threshold used for adaptive sampling that should not be exceeded by NNs for energy predictions in eV.', default=1)
    pred_parser.add_argument('--thresholds',type=float,nargs=5, help='Percentage of mean predicted by NNs taken as thresholds for adaptive sampling - first value for energy, second for forces, third for dipoles, fourth for nacs, fifth for socs. Units are [eV, eV/A, Debye, a.u. and cm-1]', default=[1,1,1,1,1])
    pred_parser.add_argument('--modelpaths',type=str,nargs="*", help='Path of stored models')
    pred_parser.add_argument('modelpath', help='Path of stored model')
    pred_parser.add_argument('--hessian', action='store_true', help='Gives back the hessian of the PES.')
    pred_parser.add_argument('--adaptive', action='store_true', default=None,help='Adaptive Sampling initializer, takes mean of models for dynamics')
    pred_parser.add_argument('--emodel2', type=str, help='Path to the second model used for energies')
    pred_parser.add_argument('--nacmodel', type=str, help='Path to the second model used for NACs',default=None)
    pred_parser.add_argument('--socmodel', type=str, help='Path to the second model used for socs',default=None)
    pred_parser.add_argument('--diss_tyro', action='store_true', help='Define bond distances at which the energy is set constant for X-H bonds.', default=None)
    pred_parser.add_argument("--finish",action="store_true",help="assume that the dynamics will only occur between S1 and S0.")
    pred_parser.add_argument('--nac_approx',type=float, nargs=3, default=[1,0.036,0.036],help='Type of NAC approximation as first value and threshold for energy gap in Hartree as second value.')
    # model-specific parsers
    CIopt_parser = argparse.ArgumentParser(add_help=False, parents=[cmd_parser])
    CIopt_parser.add_argument('--nac_approx',type=float, nargs=3, default=[1,0.036,0.036],help='Type of NAC approximation as first value and threshold for energy gap in Hartree as second value.')
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

    schnet_parser.add_argument('--n_neurons', type=int, default=100,
                              help='Number of nodes in atomic networks (default: %(default)s)')
    #######  invD  ########
    invD_parser = argparse.ArgumentParser(add_help=False, parents=[model_parser])
    invD_parser.add_argument('--n_atoms', type=int, help='Number of atoms in the molecule',
                             default="ADD NUMBER OF ATOMS WITH ARGUENT --n_atoms")
    invD_parser.add_argument('--n_layers', type=int, default=3,
                               help='Number of layers in output networks (default: %(default)s)')
    invD_parser.add_argument('--n_neurons', type=int, default=100,
                              help='Number of nodes in atomic networks (default: %(default)s)')
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
    subparser_CIopt = cmd_subparsers.add_parser('CIopt', help='Eval help',parents=[CIopt_parser])
    subparser_pred = cmd_subparsers.add_parser('pred', help='Eval help', parents=[pred_parser])

    train_subparsers = subparser_train.add_subparsers(dest='model', help='Model-specific arguments')
    train_subparsers.required = True
    train_subparsers.add_parser('invD', help='invD help', parents=[train_parser, invD_parser])
    train_subparsers.add_parser('schnet', help='SchNet help', parents=[train_parser, schnet_parser])
    train_subparsers.add_parser('wacsf', help='wACSF help', parents=[train_parser, wacsf_parser])

    eval_subparsers = subparser_eval.add_subparsers(dest='model', help='Model-specific arguments')
    eval_subparsers.required = True
    eval_subparsers.add_parser('invD', help='invD help', parents=[eval_parser, invD_parser])
    eval_subparsers.add_parser('schnet', help='SchNet help', parents=[eval_parser, schnet_parser])
    eval_subparsers.add_parser('wacsf', help='wACSF help', parents=[eval_parser, wacsf_parser])

    CIopt_subparsers = subparser_CIopt.add_subparsers(dest="mdel",help="Model-specific arguments")
    CIopt_subparsers.required = True
    CIopt_subparsers.add_parser('schnet', help='SchNet help', parents=[eval_parser,schnet_parser])
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
        if prop=="forces" and train_args.no_negative_dr == True:
            prop2="gradient"
        else:
            prop2=prop
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
                    spk.metrics.MeanAbsoluteError(prop, prop2),
                    spk.metrics.RootMeanSquaredError(prop, prop2)
                ]
        else:
          metrics += [
              spk.metrics.MeanAbsoluteError(prop, prop2),
              spk.metrics.RootMeanSquaredError(prop, prop2)
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
                #diagonal energies included for training of socs
                elif prop == "socs" and args.diagonal == True or prop=="old_socs" and args.diagonal==True:
                    #indicate true to only include the error in the diagonal for training of socs
                    #indicate false to include also the error on socs (minimum function)
                    smooth=False
                    prop_diff = schnarc.nn.diagonal_phaseloss(batch, result,prop, n_states,props_phase[2],False,False,smooth,float(tradeoffs[prop].split()[0]),1.0)
                    prop_err = prop_diff #torch.mean(prop_diff.view(-1))
                elif prop == "socs" and combined_phaseless_loss == False and args.diagonal==False:
                    prop_diff = schnarc.nn.min_loss_single_old(batch[prop], result[prop], smooth=False, smooth_nonvec=False, loss_length=False)
                    #prop_diff = schnarc.nn.min_loss_single(batch[prop], result[prop], combined_phaseless_loss, n_states, props_phase )
                    prop_err = torch.mean(prop_diff.view(-1) )
                elif prop == "dipoles" and combined_phaseless_loss == True:
                    prop_err = schnarc.nn.min_loss(batch[prop], result[prop], combined_phaseless_loss, n_states, props_phase, phase_vector_nacs, dipole=True )
                    #already spared and mean of all values
                    prop_err = torch.mean(prop_diff.view(-1))
                elif prop == "dipoles" and combined_phaseless_loss == False:
                    #ATTENTION: The permanent dipole moments contain an arbitrary sign when training with the following function
                    prop_diff = schnarc.nn.min_loss_single_old(batch[prop], result[prop],loss_length=False)
                    #use the following if the signs of the permanent dipole moments should be assigned:
                    #prop_diff = schnarc.nn.min_loss_single(batch[prop], result[prop], combined_phaseless_loss, n_states, props_phase, dipole = True )
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
                if prop=='energy' and result[prop].shape[1]==int(1) or prop=='forces' and result[prop].shape[1]==int(1) or prop=='dipoles' and result[prop].shape[1]==int(1):
                    prop_diff = batch[prop][:,0] - result[prop][:,0]
                else:
                    if prop == "forces" and args.no_negative_dr == True:
                        prop_diff = batch[prop] - result["gradient"]
                    else:
                        prop_diff= batch[prop]-result[prop]
                prop_err = torch.mean(prop_diff.view(-1) ** 2)
            err_sq = err_sq + float(tradeoffs[prop].split()[0]) * prop_err
            if args.verbose:
                print(prop, prop_err)

        return err_sq

    trainer = spk.train.Trainer(args.modelpath, model, loss, optimizer,
                                train_loader, val_loader, hooks=hooks)
    trainer.train(device)


def evaluate(args, model, train_loader, val_loader, test_loader, device, train_args):
    # Get property names from model
    if args.parallel:
        model=model.module.to(device)
        properties=model.output_modules[0].properties
    else:
        properties = model.output_modules[0].properties
    header = ['Subset']
    metrics = []
    for prop in properties:
        if prop == "forces" and train_args.no_negative_dr:
            prop2="gradient"
        else:
            prop2=prop
        header += [f'{prop}_MAE', f'{prop}_RMSE']
        if prop in schnarc.data.Properties.phase_properties:
            header += [f'{prop}_pMAE', f'{prop}_pRMSE']
            metrics += [
                schnarc.metrics.PhaseMeanAbsoluteError(prop, prop),
                schnarc.metrics.PhaseRootMeanSquaredError(prop, prop)
            ]
        else:
            metrics += [
                schnarc.metrics.MeanAbsoluteError(prop, prop2),
                schnarc.metrics.RootMeanSquaredError(prop, prop2)
            ]

    results = []
    if ('train' in args.split) or ('all' in args.split):
        logging.info('Training split...')
        results.append(['training'] + ['%.7f' % i for i in evaluate_dataset(metrics, model, train_loader, device,properties, train_args)])

    if ('validation' in args.split) or ('all' in args.split):
        logging.info('Validation split...')
        results.append(['validation'] + ['%.7f' % i for i in evaluate_dataset(metrics, model, val_loader, device,properties, train_args)])
    if ('test' in args.split) or ('all' in args.split):
        logging.info('Testing split...')
        results.append(['test'] + ['%.7f' % i for i in evaluate_dataset(metrics, model, test_loader, device,properties, train_args)])
    header = ','.join(header)
    results = np.array(results)

    np.savetxt(os.path.join(args.modelpath, 'evaluation.csv'), results, header=header, fmt='%s', delimiter=',')

def evaluate_dataset(metrics, model, loader, device,properties, train_args):
    # TODO: Adapt for SCHNARC, copy old
    for metric in metrics:
        metric.reset()

    qm_values={}
    predicted={}
    header=[]
    for batch in loader:
        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }
        result = model(batch)
        for prop in result:
            if train_args.no_negative_dr == True and prop == "forces":
                if prop in predicted:
                    predicted[prop] += [result["gradient"].cpu().detach().numpy()]
                else:
                    predicted[prop] = [result["gradient"].cpu().detach().numpy()]
            else:
                if prop in predicted:
                    predicted[prop] += [result[prop].cpu().detach().numpy()]
                else:
                    predicted[prop] = [result[prop].cpu().detach().numpy()]
        for prop in batch:
            if prop in qm_values:
                qm_values[prop] += [batch[prop].cpu().detach().numpy()]
            else:
                qm_values[prop] = [batch[prop].cpu().detach().numpy()]
                    
        for metric in metrics:
            metric.add_batch(batch, result)
    results = [
    metric.aggregate() for metric in metrics
    ]

    prediction_path = os.path.join(args.modelpath,"evaluation_values_")
    prediction_path_qm = os.path.join(args.modelpath,"evaluation_qmvalues_")
    #for p in predicted.keys():
    #    np.savez(prediction_path+"%s"%p,predicted[p])
    #    #predicted[p]=np.vstack(predicted[p])
    #for p in qm_values.keys():
    #    np.savez(prediction_path_qm+"%s"%p,qm_values[p])
    #    #qm_values[p]=np.vstack(qm_values[p])
    np.savez(prediction_path,**predicted)
    np.savez(prediction_path_qm,**qm_values)
    logging.info('Stored model predictions in {:s} ...'.format(prediction_path))

    return results


def run_prediction(model, loader, device, args):
    from tqdm import tqdm
    import numpy as np
    if args.parallel:
        model=model.module.to(device)
        properties=model.output_modules[0].properties
    else:
        properties = model.output_modules[0].properties

    predicted = {}
    qm_values = {}
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
        for prop in batch:
            if prop in qm_values:
                qm_values[prop] += [batch[prop].cpu().detach().numpy()]
            else:
                qm_values[prop] = [batch[prop].cpu().detach().numpy()]


    prediction_path = os.path.join(args.modelpath,"pred_values_")
    prediction_path_qm = os.path.join(args.modelpath,"pred_qmvalues_")
    #for p in predicted.keys():
    #    np.savez("%s%s"%(prediction_path,p),predicted[p])
    #    #predicted[p]=np.vstack(predicted[p])
    #for p in qm_values.keys():
    #    np.savez("%s%s"%(prediction_path_qm,p),qm_values[p])
    #    #qm_values[p]=np.vstack(qm_values[p])
    prediction_path = os.path.join(args.modelpath, 'predictions.npz')
    np.savez(prediction_path, **predicted)
    logging.info('Stored model predictions in {:s}...'.format(prediction_path))


def get_model(args, n_states, properties, atomref=None, mean=None, stddev=None, train_loader=None, parallelize=False,
              mode='train'):
    if args.model == 'schnet':
        representation = spk.representation.SchNet(args.features, args.features, args.interactions,
                                                   args.cutoff / units.Bohr, args.num_gaussians)

        property_output = schnarc.model.MultiStatePropertyModel(args.features, n_states, n_neurons=args.n_neurons,properties=properties,
                                                                mean=mean, stddev=stddev, atomref=atomref,
                                                                n_layers=args.n_layers, real=args.real_socs,
                                                                inverse_energy=args.inverse_nacs)
        model = spk.atomistic.AtomisticModel(representation, property_output)
    elif args.model == 'invD':
        representation = spk.representation.schnet.invD()
        n_in = (args.n_atoms*args.n_atoms-args.n_atoms)
        #neurons = number of neurons in each layer of the network; if None, it is divided by two in each layer
        property_output = schnarc.model.MultiStatePropertyModel(n_in, n_states,n_neurons=args.n_neurons, properties=properties,
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


def orca_opt(args, model, device):
    # reads QM.in and writes QM.out for optimizations with ORCA

    #read QMin
    QMin=read_QMin("QM.in")
    # get input for schnet
    schnet_inputs = QMin2schnet(QMin,device)
    #perform prediction
    schnet_outputs = model(schnet_inputs)
    for key,value in schnet_outputs.items():
        schnet_outputs[key] = value.cpu().detach().numpy()
    #transform to QM.out
    get_QMout(schnet_outputs, QMin,args)
    return

def get_QMout(schnet_outputs,QMin,args):
    from transform_prediction_QMout import QMout
    QMout(schnet_outputs,args.modelpath,args.nac_approx,QMin)
    return

def QMin2schnet(QMin,args):
    from schnetpack import Properties
    from ase import Atoms
    from schnetpack.environment import SimpleEnvironmentProvider, collect_atom_triples

    environment_provider = SimpleEnvironmentProvider()
    schnet_inputs = dict()
    molecule = Atoms(QMin['atypes'],QMin['geom'])
    schnet_inputs[Properties.Z] = torch.LongTensor(molecule.numbers.astype(np.int))
    schnet_inputs[Properties.atom_mask] = torch.ones_like(schnet_inputs[Properties.Z]).float()
    #set positions
    positions = molecule.positions.astype(np.float32)
    schnet_inputs[Properties.R] = torch.FloatTensor(positions)
    #get atom environment
    nbh_idx, offsets = environment_provider.get_environment(molecule)
    #neighbours and neighour mask
    mask = torch.FloatTensor(nbh_idx) >= 0
    schnet_inputs[Properties.neighbor_mask] = mask.float()
    schnet_inputs[Properties.neighbors] = torch.LongTensor(nbh_idx.astype(np.int)) * mask.long()
    #get cells
    schnet_inputs[Properties.cell] = torch.FloatTensor(molecule.cell.astype(np.float32))
    schnet_inputs[Properties.cell_offset]= torch.FloatTensor(offsets.astype(np.float))
    collect_triples = False
    if collect_triples is not None:
        nbh_idx_j, nbh_idx_k, offset_idx_j, offset_idx_k = collect_atom_triples(nbh_idx)
        schnet_inputs[Properties.neighbor_pairs_j] = torch.LongTensor(nbh_idx_j.astype(np.int))
        schnet_inputs[Properties.neighbor_pairs_k] = torch.LongTensor(nbh_idx_k.astype(np.int))
        schnet_inputs[Properties.neighbor_pairs_mask] = torch.ones_like(schnet_inputs[Properties.neighbor_pairs_j]).float()                                                                 # Add batch dimension and move to CPU/GPU
    for key, value in schnet_inputs.items():
        schnet_inputs[key] = value.unsqueeze(0).to(device)

    return schnet_inputs


def read_QMin(filename):
    A2Bohr = 1/0.529177211
    data = open(filename, 'r').readlines()
    QMin = {}
    natoms = int(data[0].split()[0])
    QMin['NAtoms'] = natoms
    #unit
    for line in data:
        if "angstrom" in line or "Angstrom" in line:
            angstrom = True
            break
        else:
            angstrom = False
    #states
    for line in data:
        if "states" in line:
            s=line.split()[1:]
            break
    states = [ int(i) for i in s]
    n_doublets = 0
    n_triplets = 0
    n_quartets = 0
    for index,istate in enumerate(states):
        if index == int(0):
            n_singlets = istate
        elif index == int(1):
            n_doublets = istate
        elif index == int(2):
            n_triplets = istate
        elif index == int(3):
            n_quartets = istate
    nmstates = n_singlets + 2*n_doublets + 3*n_triplets + 4*n_quartets
    QMin['n_singlets'] = n_singlets
    QMin['n_doublets'] = n_doublets
    QMin['n_triplets'] = n_triplets
    QMin['n_quartets'] = n_quartets
    QMin['n_states'] = nmstates

    #geometries
    atypes = []
    geom = []
    for curr_atom in range(natoms):
        atom_data = data[curr_atom+2].strip().split()
        curr_atype = atom_data[0]
        if angstrom == True:
            curr_geom = [float(coordinate) for coordinate in atom_data[1:4]]
            for xyz in range(3):
                curr_geom[xyz] = curr_geom[xyz] * A2Bohr
        else:
            curr_geom = [float(coordinate) for coordinate in atom_data[1:4]]
        atypes.append(curr_atype)
        geom.append(curr_geom)
    QMin['geom'] = np.array(geom)
    QMin['atypes'] = np.array(atypes)

    return QMin

if __name__ == '__main__':
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Determine the device
    device = torch.device("cuda" if args.cuda else "cpu")
    if args.mode != 'train':
        model = torch.load(os.path.join(args.modelpath, 'best_model'), map_location='cpu').to(device)
        if args.hessian == True:
            if args.parallel==False:
                for p in model.output_modules[0].output_dict.keys():
                    model.output_modules[0].output_dict[p].return_hessian = [True,1,1,1,1,False]
            else:
                for p in model.module.output_modules[0].output_dict.keys():
                    model.module.output_modules[0].output_dict[p].return_hessian = [True,1,1,1,1,False]
        else:
            if args.parallel==False:
                for p in model.output_modules[0].output_dict.keys():
                    model.output_modules[0].output_dict[p].return_hessian = [False,1,1,1,1,False]
            else:
                for p in model.module.output_modules[0].output_dict.keys():
                    model.module.output_modules[0].output_dict[p].return_hessian = [False,1,1,1,1,False]

    if args.mode == 'CIopt':
        orca_opt(args, model, device)
        sys.exit()


    if args.mode == 'pred':
        pred_data = spk.data.AtomsData(args.datapath)
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
    environment_provider = get_environment_provider(train_args,device=torch.device("cuda" if args.cuda else "cpu"))
    dataset = spk.data.AtomsData(args.datapath, environment_provider=environment_provider,collect_triples=args.model == 'wacsf')
    # Determine the number of states based on the metadata
    n_states = {}
    n_states['n_singlets'] = dataset.get_metadata("n_singlets")
    if dataset.get_metadata("n_triplets") == None:
        n_states['n_triplets']=int(0)
    else:
        n_states['n_triplets'] = dataset.get_metadata("n_triplets")
    n_states['n_states'] = n_states['n_singlets'] + n_states['n_triplets']
    if args.log == True:
        a=dataset.get_metadata()
        n_states["n_socs"] = len(a["socsindex"])
    ##activate if only one state is learned or not all
    for p in properties:
        s=tradeoffs[p].split()
    if int(s[1]) > int(0):
        n_singlets = int(s[1])
        n_triplets = int(s[2])
        n_states['n_singlets'] = n_singlets
        n_states['n_triplets'] = n_triplets
        n_states['n_states'] = n_states['n_singlets'] + n_states['n_triplets']
    n_states['states'] = dataset.get_metadata("states")
    n_states["finish"] = args.finish
    n_states["order"] = args.order
    logging.info('Found {:d} states... {:d} singlet states and {:d} triplet states'.format(n_states['n_states'],
                                                                                           n_states['n_singlets'],
                                                                                           n_states['n_triplets']))

    if args.mode == 'eval':
        split_path = os.path.join(args.modelpath, 'split.npz')
        #data_train, data_val, data_test = dataset.create_splits(*args.split, split_file=split_path)
        train_loader, val_loader, test_loader = get_loaders(args,dataset=dataset,split_path=split_path, logging=logging)

        #train_loader = spk.data.AtomsLoader(data_train, batch_size=args.batch_size, sampler=RandomSampler(data_train),
        #                                num_workers=4, pin_memory=True)
        #val_loader = spk.data.AtomsLoader(data_val, batch_size=args.batch_size, num_workers=2, pin_memory=True)

        logging.info("evaluating...")
        #test_loader = spk.data.AtomsLoader(data_test, batch_size=args.batch_size,
        #                                   num_workers=2, pin_memory=True)
        evaluate(args, model, train_loader, val_loader, test_loader, device, train_args)
        logging.info("... done!")
        exit()
     # Splits the dataset in test, val, train sets
    split_path = os.path.join(args.modelpath, 'split.npz')
 
    if args.mode == 'train':
        if args.split_path is not None:
            copyfile(args.split_path, split_path)

    logging.info('Creating splits...')
    #removed outdated loader
    #data_train, data_val, data_test = dataset.create_splits(*train_args.split, split_file=split_path)
    train_loader, val_loader, test_loader = get_loaders(args,dataset=dataset,split_path=split_path, logging=logging)
    
    # Generate loaders for training
    #train_loader = spk.data.AtomsLoader(data_train, batch_size=args.batch_size, sampler=RandomSampler(data_train),
    #                                    num_workers=4, pin_memory=True)
    #val_loader = spk.data.AtomsLoader(data_val, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    if args.transfer is not None:
        dataset_transfer = spk.data.AtomsData(args.transfer, environment_provider=environment_provider,collect_triples=args.model == 'wacsf')
        split_path2=os.path.join(args.modelpath,'split_transfer.npz')
        data_transfer, data_val_, data_test_ = dataset_transfer.create_splits(*train_args.split, split_file=split_path2)
        transfer_loader = spk.data.AtomsLoader(data_transfer,batch_size=args.batch_size,num_workers=4,pin_memory=True)
    # Determine statistics for scaling
    if args.mode == 'train':
        # Set the random seed
        spk.utils.set_random_seed(args.seed)

        logging.info('Calculate statistics...')
        mean = {}
        stddev = {}
        # Compute mean and stddev for relevant properties
        for p in properties:
            if p in schnarc.data.Properties.normalize or p=="socs" and args.log==False:
                if args.transfer is not None and args.diagonal is None:
                    mean_p,stddev_p = transfer_loader.get_statistics(p,True)
                    mean_p = mean_p[p]
                    stddev_p = stddev_p[p]
                    mean[p] = mean_p
                    stddev[p] = stddev_p
                    logging.info('{:s} MEAN: {:20.11f} STDDEV: {:20.11f}'.format(p, mean_p.detach().cpu().numpy()[0],
                                                                          stddev_p.detach().cpu().numpy()[0]))
                elif args.transfer is None and args.diagonal is None:
                    mean_p, stddev_p = train_loader.get_statistics(p, True)
                    mean_p = mean_p[p]
                    stddev_p = stddev_p[p]
                    mean[p] = mean_p
                    stddev[p] = stddev_p
                    logging.info('{:s} MEAN: {:20.11f} STDDEV: {:20.11f}'.format(p, mean_p.detach().cpu().numpy()[0],
                                                                          stddev_p.detach().cpu().numpy()[0]))
                else:
                    mean_p = None
                    stddev_p = None
                    mean[p] = mean_p
                    stddev[p] = stddev_p
            else:
                mean[p]=None
                stddev[p]=None
    else:
        mean, stddev = None, None
    #if "energy" in properties:
    #    atomref = dataset.get_atomref("energy")
    #else:
    #    atomref = None
    atomref = None
    # Construct the model.
    model = get_model(train_args,
                      n_states,
                      properties,
                      atomref=atomref,
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
        if args.log == False:
            n_socs = int(all_states * (all_states - 1)) # complex so no division by zero
        if args.diagonal:
            a=dataset.get_metadata()
            socindex = a["socsindex"]
            n_socs = int(all_states * (all_states - 1)) # complex so no division by zero
            mask_socs = np.zeros((n_socs))
            for socindex_mask,maskvalue in enumerate(socindex):
                mask_socs[maskvalue]=1.0
            n_states["mask_socs"]=mask_socs
        if args.log == True:
            a=dataset.get_metadata()
            n_socs = len(a["socsindex"])
            socindex = a["socsindex"]
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

        props_phase=[n_phases,batch_size,device,phase_pytorch,n_socs, all_states, socs_phase_matrix_1, socs_phase_matrix_2, diag_phase_matrix_1, diag_phase_matrix_2,mean,stddev]
        train(args, model, tradeoffs, train_loader, val_loader, device, n_states, props_phase)
        logging.info("...training done!")
