from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from datasets.dataset_generic_npy import Generic_MIL_Dataset
from datasets.dataset_nic import Generic_MIL_Dataset as NIC_MIL_Dataset
from datasets.dataset_nic_ovarian import Generic_MIL_Dataset_ovarian as NIC_MIL_Dataset_ovarian
import h5py
from utils.eval_utils import *

# CUDA_VISIBLE_DEVICES=7 python eval.py --drop_out --k 5 --models_exp_code renal_subtyping_CLAMMB_ctranspath_1512_5fold_s1 --save_exp_code renal_subtyping_CLAMMB_ctranspath_1512_5fold_cv --task renal_subtype --model_type clam_mb --results_dir results --data_root_dir /home3/gzy/Renal/

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default='/home5/gzy/Cameylon/feature_resnet',
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default='renal_subtyping_CLAMMB_noinst_res50_1512_5fold_s1',
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, default='clam_sb', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--drop_out', action='store_true', default=False, 
                    help='whether model uses dropout')

parser.add_argument('--k', type=int, default=5, help='number of folds (default: 5)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['renal_subtype','camelyon','lung_subtype','renal_subtype_yfy'])
parser.add_argument('--fea_dim', type=int, default=1024,
                     help='the original dimensions of patch embedding')
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=True, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')

### NICWSS specific options
parser.add_argument('--only_cam', action='store_true', default=False,
                    help='if only cam, without PCM module')
parser.add_argument('--inst_rate', type=float, default=0.01,
                    help='drop_rate for drop_with_score')
parser.add_argument('--b_rv', type=float, default=1.0, 
                    help='parameter to balance the cam and cam_rv')
parser.add_argument('--w_cls', type=float, default=1.0, 
                    help='loss function weight for classification')
parser.add_argument('--w_er', type=float, default=1.0, 
                    help='loss function weight for Equivarinat loss')
parser.add_argument('--w_ce', type=float, default=1.0, 
                    help='loss function weight for conditional entropy')

args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.save_exp_code=args.models_exp_code
args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

conf_file = 'experiment_' + '_'.join(args.models_exp_code.split('_')[:-1])+'.txt'
fr = open(os.path.join(args.models_dir, conf_file),'r')
conf = eval(fr.read())
fr.close()

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir
    
assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

# update configs
args.task=conf['task']
if 'fea_dim' in conf:
    args.fea_dim=conf['fea_dim']
args.model_type=conf['model_type']
args.model_size=conf['model_size']
args.drop_out=conf['use_drop_out']
if 'no_inst_cluster' in conf:
    args.no_inst_cluster=conf['no_inst_cluster']
if 'inst_loss' in conf:
    args.inst_loss=conf['inst_loss']
    
if args.model_type in ['clam_sb', 'clam_mb']:
    if 'subtyping' in conf:
        args.subtyping=conf['subtyping']
    if 'bag_weight' in conf:
        args.bag_weight=conf['bag_weight']
    if 'B' in conf:
        args.B=conf['B']
elif args.model_type in ['nic', 'nicwss']:
    args.only_cam=conf['only_cam']
    args.b_rv = conf['b_rv']
    args.w_cls = conf['w_cls']
    args.w_er = conf['w_er']
    args.w_ce = conf['w_ce']
    

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

if args.model_type in ['nic', 'nicwss']:
    settings.update({'only_cam': args.only_cam,
                     'b_rv': args.b_rv,
                     'w_cls': args.w_cls,
                     'w_er': args.w_er,
                     'w_ce': args.w_ce})



with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)

dataset_params = {
    'csv_path' : 'dataset_csv/renal_subtyping_npy.csv',
    'data_dir' : args.data_root_dir,
    'data_mag' :'1_512',
    'shuffle' : False, 
    'seed' : 10, 
    'print_info': True,
    'label_dict' : {'ccrcc':0, 'prcc':1, 'chrcc':2},
    'patient_strat': False,
    'ignore': []
}


if args.task == 'renal_subtype':
    args.n_classes=3
    args.patch_size=2048
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/renal_subtyping_npy.csv',
                                data_dir = args.data_root_dir,
                                data_mag = '1_512',
                                shuffle = False, 
                                seed = 10, 
                                print_info = True,
                                label_dict = {'ccrcc':0, 'prcc':1, 'chrcc':2},
                                patient_strat= False,
                                ignore=[])
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        assert args.subtyping

# ny conch feature
elif args.task == 'kica_subtyping':
    args.n_classes=2
    args.patch_size=512
    dataset_params['csv_path'] = 'dataset_csv/kica_subtyping_npy.csv'
    dataset_params['label_dict'] = {'chromophobe type':0, 'Papillary adenocarcinoma': 1}
    dataset_params['data_mag'] = '10x512'
    if args.model_type in ['nicwss', 'nic']:
        dataset = NIC_MIL_Dataset(**dataset_params)
    else:
        dataset = Generic_MIL_Dataset(**dataset_params)

elif args.task == 'kica_staging':
    args.n_classes=2
    args.patch_size=512
    dataset_params['csv_path'] = 'dataset_csv/kica_staging_npy.csv'
    dataset_params['label_dict'] = {'late':0, 'early': 1}
    dataset_params['data_mag'] = '10x512'
    if args.model_type in ['nicwss', 'nic']:
        dataset = NIC_MIL_Dataset(**dataset_params)
    else:
        dataset = Generic_MIL_Dataset(**dataset_params)

elif args.task == 'renal_subtype_yfy':
    args.n_classes=3
    args.patch_size=1024
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/renal_subtyping_yfy_npy.csv',
                                data_dir = args.data_root_dir,
                                data_mag = '0_1024',
                                shuffle = False, 
                                seed = 10, 
                                print_info = True,
                                label_dict = {'ccrcc':0, 'prcc':1, 'chrcc':2},
                                patient_strat= False,
                                ignore=[])
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        assert args.subtyping 
elif args.task == 'camelyon':
    args.n_classes=2
    args.patch_size=512
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/camelyon_npy.csv',
                                data_dir = os.path.join(args.data_root_dir),
                                data_mag = '0_512',
                                shuffle = False, 
                                seed = 10, 
                                print_info = True,
                                label_dict = {'normal':0, 'tumor':1},
                                patient_strat= False,
                                ignore=[])
elif args.task == 'lung_subtype':
    args.n_classes=2
    args.patch_size=2048
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/lung_subtyping_npy.csv',
                                data_dir = os.path.join(args.data_root_dir),
                                data_mag = '1_512',
                                shuffle = False, 
                                seed = 10, 
                                print_info = True,
                                label_dict = {'luad':0, 'lusc':1},
                                patient_strat= False,
                                ignore=[])

else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint_best.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    all_inst_auc=[]
    all_inst_acc=[]
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
        model, patient_results, test_error, auc, df, df_inst  = eval_(split_dataset, args, ckpt_paths[ckpt_idx])
        all_results.append(all_results)
        all_auc.append(auc[0])
        all_inst_auc.append(auc[1])
        all_inst_acc.append(auc[2])
        all_acc.append(1-test_error)
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)
        df_inst.to_csv(os.path.join(args.save_dir, 'fold_{}_inst.csv'.format(folds[ckpt_idx])), index=False)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc,'test_iauc':all_inst_auc,'test_iacc':all_inst_acc})
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))
