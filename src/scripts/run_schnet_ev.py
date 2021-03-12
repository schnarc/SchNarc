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
from schnetpack.utils.script_utils.settings import get_environment_provider
from schnetpack.utils import get_dataset
import schnetpack as spk
import schnet_ev
from schnet_ev.data import Eigenvalue_properties, Delta_EV_properties, Charges
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
    cmd_parser.add_argument("--aggregation_mode",type=str,default="build",choices=["avg","build"])
    cmd_parser.add_argument('--environment_provider',type=str, default="simple", choices=["simple","ase","torch"],help="Environment provider for dataset. (default: %(default)s)",)
    cmd_parser.add_argument('--force_mask_index',help='Atom number of excluded atom for force training', type=int)
    cmd_parser.add_argument('--force_mask', help= 'Only train on forces for specific atoms', action="store_true")
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
                              default=25)
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
    train_parser.add_argument('--hamiltonian', action='store_true',
                              help='If True, approximate diabatic Hamiltonian using partial charges without diagonalization to train on eigenvalues.')
    train_parser.add_argument('--diabatic', action='store_true',
                              help='If True, approximate diabatic Hamiltonian and diagonalize to train on eigenvalues.',default=True)
    train_parser.add_argument('--phase_loss', action='store_true', help='Use special loss, which ignores phase.')
    train_parser.add_argument('--inverse_nacs', action='store_true', help='Weight NACs with inverse energies.')
    train_parser.add_argument('--smooth_loss', action='store_true', help='Use phase independent min loss.')
    train_parser.add_argument('--mixed_loss', action='store_true', help='Use phase independent min loss.')
    train_parser.add_argument('--diagonal', action='store_true', help='Train SOCs via diagonal elements. Must be included in the training data base', default=None)
    train_parser.add_argument('--L1', action='store_true', help='Use L1 norm')
    train_parser.add_argument('--Huber', action='store_true', help='Use L1 norm')

    ## evaluation
    eval_parser = argparse.ArgumentParser(add_help=False, parents=[cmd_parser])
    eval_parser.add_argument('datapath', help='Path / destination of MD17 dataset directory')
    eval_parser.add_argument('modelpath', help='Path of stored model')
    eval_parser.add_argument('--split', help='Evaluate on trained model on given split',
                             choices=['train', 'validation', 'test', 'all'], default=['all'], nargs='+')
    eval_parser.add_argument('--hessian', action='store_true', help='Gives back the hessian of the PES.')
    eval_parser.add_argument('--hamiltonian', action='store_true',
                              help='If True, approximate diabatic Hamiltonian using partial charges without diagonalization to train on eigenvalues.')
    eval_parser.add_argument('--diabatic', action='store_true',
                              help='If True, approximate diabatic Hamiltonian and diagonalize to train on eigenvalues.',default=True)

    pred_parser = argparse.ArgumentParser(add_help=False, parents=[cmd_parser])
    pred_parser.add_argument('datapath', help='Path / destination of MD17 dataset directory')
    pred_parser.add_argument('--print_uncertainty',action='store_true', help='Print uncertainty of each property',default=False)
    pred_parser.add_argument('--diabatic', action='store_true',
                              help='If True, approximate diabatic Hamiltonian and diagonalize to train on eigenvalues.',default=True)
    pred_parser.add_argument('--thresholds',type=float,nargs=5, help='Percentage of mean predicted by NNs taken as thresholds for adaptive sampling - first value for energy, second for forces, third for dipoles, fourth for nacs, fifth for socs', default=None)
    pred_parser.add_argument('--modelpaths',type=str, help='Path of stored models')
    pred_parser.add_argument('modelpath', help='Path of stored model')
    pred_parser.add_argument('--hessian', action='store_true', help='Gives back the hessian of the PES.')
    pred_parser.add_argument('--adaptive',type=float, nargs=1,help='Adaptive Sampling initializer + float for reducing the step size')
    pred_parser.add_argument('--nac_approx',type=float, nargs=3, default=[1,0.018,0.036],help='Type of NAC approximation as first value and threshold for energy gap in Hartree as second value.')
    # model-specific parsers
    opt_parser = argparse.ArgumentParser(add_help=False, parents=[cmd_parser])
    opt_parser.add_argument('--nac_approx',type=float, nargs=3, default=[1,0.018,0.036],help='Type of NAC approximation as first value and threshold for energy gap in Hartree as second value.')
    model_parser = argparse.ArgumentParser(add_help=False)
 
    #######  SchNet  #######
    schnet_parser = argparse.ArgumentParser(add_help=False, parents=[model_parser])
    schnet_parser.add_argument('--features', type=int, help='Size of atom-wise representation (default: %(default)s)',
                               default=256)
    schnet_parser.add_argument("--cutoff_function",help="Functional form of the cutoff", choices=["hard","cosine","mollifier"],default="cosine")
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
    subparser_opt = cmd_subparsers.add_parser('opt', help='Eval help',parents=[opt_parser])
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

    opt_subparsers = subparser_opt.add_subparsers(dest="mdel",help="Model-specific arguments")
    opt_subparsers.required = True
    opt_subparsers.add_parser('schnet', help='SchNet help', parents=[eval_parser,schnet_parser])
    return main_parser


def train(args, model, tradeoffs, train_loader, val_loader, device, n_orb,mean,stddev,offset):
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
      if prop == "eigenvalues_active_forces" or prop == "eigenvalues_forces":
          pass
      else:
          metrics += [
              schnet_ev.metrics.MeanAbsoluteError(prop, prop),
              schnet_ev.metrics.RootMeanSquaredError(prop, prop)
          ]
    if args.logger == 'csv':
        logger = spk.train.CSVHook(os.path.join(args.modelpath, 'log'),
                                   metrics, every_n_epochs=args.log_every_n_epochs)
        hooks.append(logger)
    elif args.logger == 'tensorboard':
        logger = spk.train.TensorboardHook(os.path.join(args.modelpath, 'log'),
                                           metrics, every_n_epochs=args.log_every_n_epochs)
        hooks.append(logger)
    n_act = n_orb['n_act']
    # Automatically construct loss function based on tradoffs
    def loss(batch, result):
        err_sq = 0.0
        if args.verbose:
            print('===================')
        for prop in tradeoffs:
            if prop == 'energy' or prop == 'forces' or prop == 'homo_lumo':
                prop_diff = batch[prop] - result[prop]
                prop_err = torch.mean(prop_diff.view(-1) ** 2)
                err_sq = err_sq + float(tradeoffs[prop]) * prop_err
            if prop in Eigenvalue_properties.properties or prop in Delta_EV_properties.properties :
                if "eigenvalues_active_forces" in tradeoffs or "eigenvalues_forces" in tradeoffs:
                    result['ev_all']= result[prop]
                    result[prop] = result[prop][prop]
                #results are ordered in descending order
                if args.smooth_loss == True:
                     prop_diff =torch.normal(batch[prop],0.01)-torch.normal(result[prop],0.01)
                     prop_err = torch.mean(prop_diff.view(-1)**2)
                     err_sq = err_sq+float(tradeoffs[prop])*prop_err
                elif args.hamiltonian == True and prop == "eigenvalues":
                     #every vector has a different size - masked for every batch different the mask however is max. length and diff. for every molecule
                     prop_diff = (batch[prop][:,:result[prop].shape[1]]- result[prop]) * batch["eigenvalues_all_mask"][:,:result[prop].shape[1]]
                     prop_err = torch.mean(prop_diff.view(-1)**2)
                     err_sq = err_sq+float(tradeoffs[prop])*prop_err     
                elif args.hamiltonian == True and prop != "eigenvalues":
                     prop_diff_ev = (batch[prop][:,:result[prop].shape[1]] - result[prop])*batch["eigenvalues_active_mask"][:,:result[prop].shape[1]]
                     prop_err_ev = torch.mean(prop_diff_ev.view(-1)**2)
                     prop_err_charges=0
                     #get charges
                     if "hirshfeld_pbe" in tradeoffs:
                         q_r=torch.sum((result['coeff']**2)[:,:batch['n_elec_atoms'].shape[1],:]*batch["occupation"][:,None,:result[prop].shape[1]],dim=2)
                         result["hirshfeld_pbe"] = q_r -batch['n_elec_atoms']
                         prop_diff_charges = batch["hirshfeld_pbe"] - result["hirshfeld_pbe"]
                         prop_err_charges += torch.mean(prop_diff_charges.view(-1)**2)*tradeoffs["hirshfeld_pbe"]
                     if "hirshfeld_pbe0" in tradeoffs:
                         q_r=torch.sum((result['coeff']**2)[:,:batch['n_elec_atoms'].shape[1],:]*batch["occupation"][:,None,:result[prop].shape[1]],dim=2)
                         result["hirshfeld_pbe0"]= q_r -batch['n_elec_atoms']
                         prop_diff_charges = batch["hirshfeld_pbe0"] - result["hirshfeld_pbe0"]
                         prop_err_charges += torch.mean(prop_diff_charges.view(-1)**2)*tradeoffs["hirshfeld_pbe0"]
                     err_sq = err_sq+float(tradeoffs[prop])*(prop_err_ev)+prop_err_charges
                elif prop in Delta_EV_properties.properties:
                    prop_diff = torch.abs(batch[prop] - result[prop])*batch["eigenvalues_active_mask"]
                    prop_err = torch.mean(prop_diff.view(-1)**2)
                    err_sq = err_sq  + float(tradeoffs[prop])*prop_err
                else:
                    prop_diff = torch.abs(batch[prop] - result[prop])#*batch["eigenvalues_active_mask"]
                    prop_err = torch.mean(prop_diff.view(-1)**2)
                    err_sq = err_sq  + float(tradeoffs[prop])*prop_err
                if args.mixed_loss == True:
                    #outer subtraction
                    deltaE = torch.abs(batch[prop][...,None] - batch[prop][...,None,:])
                    #make diabatic couplings prop to 1/delta E Delta function 
                    mask= batch["%s_mask"%prop][...,None]*  batch["%s_mask"%prop][...,None,:]
                    prop_diff=(torch.abs(torch.triu(torch.normal((1/deltaE),0.1),1))*0.01-torch.abs(torch.triu((result['ev_all']['ev_diabatic']),1)))*mask
                    #print(mask,torch.triu(torch.normal((1/deltaE),0.1),1)*0.01,torch.triu((result['ev_all']['ev_diabatic'])))
                    prop_err2 = torch.mean(prop_diff.view(-1) **2 )
                    #diag forces
                    #print(result['ev_all']['ev_diab_forces'][0].shape)
                    #import numpy as np
                    #a=result['ev_all']['ev_diab_forces'][0].to('cpu').detach().numpy()
                    #a=a.reshape((3,3,7,7))
                    #u,e=np.linalg.eigh(a)
                    #print(e.shape,u.shape)
                    #multiply with properr the tradeoff so that the error gets smaller when the iegenvalues get more accurate - was not good
                    err_sq = err_sq + float(tradeoffs['eigenvalues_active_forces'])  *prop_err2
            if prop == "occupation":
                prop_diff = batch[prop] - result[prop]
                prop_err = torch.mean(prop_diff.view(-1)**2)
                err_sq = err_sq + float(tradeoffs[prop])*prop_err 
            if prop == "eigenvalues_active_forces" or prop == "eigenvalues_forces" or prop in Charges.properties:
                #batch[prop]=result['ev_all'][prop]
                pass #continue
            if prop == 'occ_orb':
                #if args.mixed_loss==True:
                #    #error on occupied orbitals
                #    prop_diff_occ_orb = batch[prop] - result[prop]
                #    prop_err_occ_orb = torch.mean(prop_diff_occ_orb.view(-1)**2)
                #    err_sq = err_sq + float(tradeoffs[prop])*prop_err_occ_orb/2 
                #    #additionally include the sum of the errors
                #    sum_occ_orb_NN = torch.sum(result[prop],1)*2
                #    prop_diff_sum_orbE = batch['sum_occ_orb'] - sum_occ_orb_NN
                #    prop_err_sum_orbE = torch.mean(prop_diff_sum_orbE.view(-1)**2)
                #    err_sq = err_sq + float(tradeoffs[prop])*prop_err_sum_orbE/2
                #    #additionally try to fit the total energy and force by taking the sum of occ.orb.energies and delta E
                #    total_E_NN = sum_occ_orb_NN + result['delta_E']
                #    total_F_NN = result['delta_E_forces']
                #    for batch_value in range(result['energy'].shape[0]):
                #        total_F_NN[batch_value][0] += torch.sum(result['occ_orb_forces'][batch_value],0)
                #    prop_diff_totE = batch['energy'] - total_E_NN
                #    prop_diff_totF = batch['forces'] - total_F_NN
                #    prop_err_totE = torch.mean(prop_diff_totE.view(-1)**2)
                #    prop_err_totF = torch.mean(prop_diff_totF.view(-1)**2)
                #    err_sq = err_sq + float(tradeoffs['energy']) * prop_err_totE + float(tradeoffs['forces']) * prop_err_totF
                if args.mixed_loss == True:
                    #error of eigenvalues based on distribution function
                    prop_diff_occ_orb =torch.normal(batch[prop],stddev[prop])-torch.normal(result[prop],stddev[prop])
                    prop_err_occ_orb = torch.mean(prop_diff_occ_orb.view(-1)**2)
                    err_sq = err_sq+float(tradeoffs[prop])*prop_err_occ_orb
                else:
                    prop_diff_occ_orb = batch[prop] - result[prop]
                    prop_err_occ_orb = torch.mean(prop_diff_occ_orb.view(-1)**2)
                    err_sq = err_sq + float(tradeoffs[prop])*prop_err_occ_orb
                    #print("err occ", prop_err_occ_orb)
            if prop == "unocc_orb":
                n_unocc=n_orb['n_unocc']
                prop_diff = batch[prop][:2] - result[prop][:2]
                prop_err = torch.mean(prop_diff.view(-1)**2)
                err_sq = err_sq + float(tradeoffs[prop])*prop_err
                #print(prop_err)
            if args.verbose:
                print(prop, prop_err)

        return err_sq

    trainer = spk.train.Trainer(args.modelpath, model, loss, optimizer,
                                train_loader, val_loader, hooks=hooks)
    trainer.train(device)


def evaluate(args, model, train_loader, val_loader, test_loader, device):
    # Get property names from model
    if args.parallel:
        model=model.module.to(device)
        properties=model.output_modules[0].properties
    else:
        properties = model.output_modules[0].properties
    header = ['Subset']
    metrics = []
    for prop in properties:
      if prop == "eigenvalues_active_forces" or prop== "eigenvalues_forces":
        pass
      else:
        header += [f'{prop}_MAE', f'{prop}_RMSE']
        metrics += [
                spk.metrics.MeanAbsoluteError(prop, prop),
                spk.metrics.RootMeanSquaredError(prop, prop)
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

    qm_values={}
    predicted={}
    batchs={}       
    header=[]
    results2={}
    """for batch in loader:
        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }
        result = model(batch)
        for prop in result:
            if prop =="eigenvalues_active": 
              if prop in result:
                result['ev_all'] += result[prop]    
                result[prop] += result[prop][prop]
              else:
                result['ev_all'] = result[prop]
                result[prop]=result[prop][prop]"""
    #do only 1000 batches for testing
    #index_=-1
    for batch in loader:
    #    index_ +=1
    #    if index_==int(1000):
    #       break
        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }
        result = model(batch)
        for prop in result:
          if prop == "eigenvalues_active": 
              if "eigenvalues_active_forces" in properties:
                if prop in predicted:
                    predicted[prop] += [(result[prop][prop]*batch["%s_mask"%prop]).cpu().detach().numpy()]
                else:
                    predicted[prop] = [(result[prop][prop]*batch["%s_mask"%prop]).cpu().detach().numpy()]
              else:
                if prop in predicted:
                    predicted[prop] += [(result[prop]*batch["%s_mask"%prop]).cpu().detach().numpy()] 
                else:                                                                               
                    predicted[prop] = [(result[prop]*batch["%s_mask"%prop]).cpu().detach().numpy()] 
          if prop =="eigenvalues" or prop in Eigenvalue_properties.properties:
                if prop in predicted:
                    predicted[prop] += [(result[prop]*batch["%s_mask"%prop]).cpu().detach().numpy()]
                else:
                    predicted[prop] = [(result[prop]*batch["%s_mask"%prop]).cpu().detach().numpy()]
          if prop == "delta_eigenvalues_pbe0_gbw" or prop=="delta_eigenvalues_pbe0_h2o":
            if prop in predicted:
                predicted[prop] += [(result[prop]*batch["eigenvalues_gbw_mask"]).cpu().detach().numpy()]
            else:
                predicted[prop] = [(result[prop]*batch["eigenvalues_gbw_mask"]).cpu().detach().numpy()]
          if prop == "energy":
            if prop in predicted:
                predicted[prop] += [(result[prop]).cpu().detach().numpy()]
            else:
                predicted[prop] = [(result[prop]).cpu().detach().numpy()]
 
        for prop in result:
          if prop =="eigenvalues_active":
                if prop in qm_values:
                    qm_values[prop] += [(batch[prop]*batch["%s_mask"%prop]).cpu().detach().numpy()]
                    qm_values["%s_mask"%prop] +=[batch["%s_mask"%prop].cpu().detach().numpy()]
                else:
                    qm_values[prop] = [(batch[prop]*batch["%s_mask"%prop]).cpu().detach().numpy()]
                    qm_values["%s_mask"%prop] = [batch["%s_mask"%prop].cpu().detach().numpy()]
          if prop == "delta_eigenvalues_pbe0_gbw" or prop =="delta_eigenvalues_pbe0_h2o":
                if prop in qm_values:
                    qm_values[prop] += [(batch[prop]*batch["eigenvalues_gbw_mask"]).cpu().detach().numpy()]
                    qm_values["eigenvalues_gbw_mask"] +=[batch["eigenvalues_gbw_mask"].cpu().detach().numpy()]
                else:
                    qm_values[prop] = [(batch[prop]*batch["eigenvalues_gbw_mask"]).cpu().detach().numpy()]
                    qm_values["eigenvalues_gbw_mask"] = [batch["eigenvalues_gbw_mask"].cpu().detach().numpy()]
 
          if prop =="eigenvalues" or prop in Eigenvalue_properties.properties:
                if prop in qm_values:
                    qm_values[prop] += [(batch[prop]).cpu().detach().numpy()]
                else:
                    qm_values[prop] = [(batch[prop]).cpu().detach().numpy()]
          if prop == "energy":
            if prop in qm_values:
                qm_values[prop] += [(batch[prop]).cpu().detach().numpy()]
            else:
                qm_values[prop] = [(batch[prop]).cpu().detach().numpy()]
        for metric in metrics:
          if "eigenvalues_active" in result:
              prop="eigenvalues_active"
              batch[prop]=batch[prop]*batch["%s_mask"%prop]
              if "eigenvalues_active_forces" in properties:
                  result[prop][prop] = result[prop][prop]*batch["%s_mask"%prop]
                  metric.add_batch(batch,result[prop])
              else:
                  result[prop]=  result[prop]#*batch["%s_mask"%prop][:len(result[prop][0])]
                  metric.add_batch(batch,result)
          else:
              metric.add_batch(batch,result)
    results = [
    metric.aggregate() for metric in metrics
    ]
    for prop in result:
        if prop == "eigenvalues_active" or prop=="eigenvalues":
            if "eigenvalues_active_forces" in properties:
                predicted['diabatic_H'] = [result[prop]['ev_diabatic'].cpu().detach().numpy()]
                #predicted['eigenvalues_active_forces'] = [result[prop]['eigenvalues_active_forces'].cpu().detach().numpy()]
    for p in predicted.keys():
        predicted[p]=np.vstack(predicted[p])
        if p in qm_values.keys():
            qm_values[p]=np.vstack(qm_values[p])
    if "eigenvalues_active_mask" in qm_values.keys():
        qm_values['eigenvalues_active_mask'] = np.array(qm_values["eigenvalues_active_mask"])
    if "eigenvalues_gbw_mask" in qm_values.keys():
        qm_values["eigenvalues_gbw_mask"] = np.array(qm_values["eigenvalues_gbw_mask"])
    if "eigenvalues_mask" in qm_values.keys():
        qm_values["eigenvalues_mask"] = np.array(qm_values["eigenvalues_mask"])
    prediction_path = os.path.join(args.modelpath,"evaluation_values.npz")
    prediction_path_qm = os.path.join(args.modelpath,"evaluation_qmvalues.npz")
    np.savez(prediction_path,**predicted)
    np.savez(prediction_path_qm,**qm_values)
    logging.info('Stored model predictions in {:s} ...'.format(prediction_path))

    return results


def run_prediction(model, loader, device, args):
    from tqdm import tqdm
    import numpy as np

    predicted = {}
    qm_values = {}
    for batch in tqdm(loader, ncols=120):
        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }
        if device != "cuda" and args.parallel==True:
	        result = model.module(batch)
        else:
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


    for p in predicted.keys():
        predicted[p] = np.vstack(predicted[p])
    #for p in qm_values.keys():
    #    qm_values[p]=np.vstack(qm_values[p])
    #prediction_path_qm = os.path.join(args.modelpath,"evaluation_qmvalues.npz")
    #np.savez(prediction_path_qm,**qm_values)
    prediction_path = os.path.join(args.modelpath, 'predictions.npz')
    np.savez(prediction_path, **predicted)
    logging.info('Stored model predictions in {:s}...'.format(prediction_path))


def get_model(args, n_orb, properties, atomref=None, mean=None, stddev=None, train_loader=None, parallelize=False,
              mode='train'):
    if args.model == 'schnet':
        representation = spk.representation.SchNet(args.features, args.features, args.interactions,
                                                   args.cutoff / units.Bohr, args.num_gaussians)

        property_output = schnet_ev.model.MultiStatePropertyModel(args.features, n_orb,n_neurons=args.n_neurons, properties=properties,
                                                                mean=mean, stddev=stddev, atomref=atomref,
                                                                n_layers=args.n_layers, diabatic=args.diabatic,hamiltonian = args.hamiltonian,
                                                                inverse_energy=args.inverse_nacs, force_mask = args.force_mask, force_mask_index = args.force_mask_index)

        model = spk.atomistic.AtomisticModel(representation, property_output)


    elif args.model == 'invD':
        representation = spk.representation.schnet.invD()
        n_in = (args.n_atoms*args.n_atoms-args.n_atoms)
        #neurons = number of neurons in each layer of the network; if None, it is divided by two in each layer
        property_output = schnet_ev.model.MultiStatePropertyModel(n_in, n_orb,n_neurons=args.n_neurons, properties=properties,
                                                                mean=mean, stddev=stddev, atomref=atomref,
                                                                n_layers=args.n_layers, diabatic=args.diabatic,hamiltonian = args.hamiltonian,
                                                                inverse_energy=args.inverse_nacs, force_mask=args.force_mask, force_mask_index = args.force_mask_index)

        model = spk.atomistic.AtomisticModel(representation, property_output)

    else:
        raise ValueError('Unknown model class:', args.model)

    if parallelize:
        model = nn.DataParallel(model)

    logging.info("The model you built has: %d parameters" % spk.utils.count_params(model))

    return model

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

    #environment_provider = SimpleEnvironmentProvider()
    environment_provider = get_environment_provider(args,device=device)
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
    n_unocc = 0
    n_quartets = 0
    for index,istate in enumerate(states):
        if index == int(0):
            n_occ = istate
        elif index == int(1):
            n_doublets = istate
        elif index == int(2):
            n_unocc = istate
        elif index == int(3):
            n_quartets = istate
    nmstates = n_occ + 2*n_doublets + 3*n_unocc + 4*n_quartets
    QMin['n_occ'] = n_occ
    QMin['n_doublets'] = n_doublets
    QMin['n_unocc'] = n_unocc
    QMin['n_quartets'] = n_quartets
    QMin['n_orb'] = nmstates

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


def get_deltaE(model):
    return

if __name__ == '__main__':
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Determine the device
    device = torch.device("cuda" if args.cuda else "cpu")
    if args.mode != 'train':
        model = torch.load(os.path.join(args.modelpath, 'best_model'), map_location='cpu').to(device)
        if args.hessian == True:
            model.output_modules[0].output_dict['energy'].return_hessian = [True,1,1,1,1]
        else:
            if args.parallel==False:
                for p in  model.output_modules[0].output_dict.keys():
                    model.output_modules[0].output_dict[p].return_hessian = [False,1,1]
            else:
                for p in model.module.output_modules[0].output_dict.keys():
                    model.module.output_modules[0].output_dict[p].return_hessian = [False,1,1]

    if args.mode == 'opt':
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
            schnet_ev.utils.generate_default_tradeoffs(tradeoff_file)
            tradeoffs = schnet_ev.utils.read_tradeoffs(tradeoff_file)
        else:
            tradeoffs = schnet_ev.utils.read_tradeoffs(args.tradeoffs)
            schnet_ev.utils.save_tradeoffs(tradeoffs, tradeoff_file)
    else:
        train_args = spk.utils.read_from_json(jsonpath)
        tradeoff_file = os.path.join(args.modelpath, 'tradeoffs.yaml')
        tradeoffs = schnet_ev.utils.read_tradeoffs(tradeoff_file)

    # Determine the properties to load based on the tradeoffs
    properties = [p for p in tradeoffs]
    if "eigenvalues_forces" in properties:
        tradeoffs['eigenvalues_active_forces']=tradeoffs['eigenvalues_forces']
    # Read and process the data using the properties found in the tradeoffs.
    logging.info('Loading {:s}...'.format(args.datapath))
    environment_provider = get_environment_provider(train_args,device=torch.device("cuda" if args.cuda else "cpu"))
    #previous function:
    dataset = spk.data.AtomsData(args.datapath, environment_provider=environment_provider, collect_triples=args.model == 'wacsf')
    # Determine the number of states based on the metadata
    n_orb = {}
    if "mean_energy" in dataset.get_metadata():
        mean_energy = dataset.get_metadata('mean_energy')
    else:
        mean_energy = int(0)
    n_orb['mean_energy']=mean_energy
    n_orb['n_act'] = int(1)
    offset=0
    for prop in properties:
        if prop in Eigenvalue_properties.properties or prop in Delta_EV_properties.properties:

        # if "eigenvalues" in properties or "eigenvalues_active" in properties or "eigenvalues_pbe0" in properties or "eigenvalues_pbe" in properties or "delta_eigenvalues_gbw" in properties or "eigenvalues_gbw" in properties:
            n_orb['n_act'] = dataset.get_metadata('active')
            if dataset.get_metadata("active_all"):
                n_orb["active_all"]= dataset.get_metadata("active_all")
            if dataset.get_metadata('offset'):
                offset = dataset.get_metadata('offset')
            else:
                offset = float(0.0)
    ##activate if only one state is learned or not all
    # only needed if unocc and occ orbitals are trained separately 
    n_orb['n_occ']=int(1)
    n_orb['n_unocc']=int(0)
    n_orb['n_orb']=n_orb['n_occ']+n_orb['n_unocc']
    n_orb['offset']=offset
    n_orb["aggregation_mode"]=args.aggregation_mode
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
            if p in schnet_ev.data.Properties.normalize:
                mean_p, stddev_p = train_loader.get_statistics(p, True)
                mean_p = mean_p[p]
                stddev_p = stddev_p[p]
                mean[p] = mean_p
                stddev[p] = stddev_p
                logging.info('{:s} MEAN: {:20.11f} STDDEV: {:20.11f}'.format(p, mean_p.detach().cpu().numpy()[0],
                                                                             stddev_p.detach().cpu().numpy()[0]))
            else:
                mean[p], stddev[p] = None, None

    # Construct the model.
    model = get_model(train_args,
                      n_orb,
                      properties,
                      mean=mean,
                      stddev=stddev,
                      train_loader=train_loader,
                      parallelize=args.parallel,
                      mode=args.mode
                      ).to(device)

    if args.mode == 'train':
        logging.info("training...")
        batch_size = args.batch_size
        train(args, model, tradeoffs, train_loader, val_loader, device, n_orb,mean,stddev,offset)
        logging.info("...training done!")

