from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train

from datasets.dataset_generic_npy import Generic_MIL_Dataset
from datasets.dataset_generic_h5 import Generic_H5_MIL_Dataset
from datasets.dataset_generic_npy_gastric_esd import Generic_MIL_Dataset as Generic_MIL_Dataset_gastric_esd


# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    all_test_recall = []
    all_test_iauc = []
    all_val_iauc = []
    folds = np.arange(start, end)
    
    for i in folds:
            
        seed_torch(args.seed)
        if args.model_type in ['clam_sb', 'clam_mb', 'PAMIL']:
            train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
            datasets = (train_dataset, val_dataset, test_dataset)
            results, test_auc, val_auc, test_acc, val_acc, test_recall, test_iauc, val_iauc = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        all_test_recall.append(test_recall)
        all_test_iauc.append(test_iauc)
        all_val_iauc.append(val_iauc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'test_recall': all_test_recall,'val_acc' : all_val_acc,
        'test_iauc':all_test_iauc, 'val_iauc': all_val_iauc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--task', type=str)
parser.add_argument('--model_type', type=str, choices=['wsod', 'wsod_nic','wsod_nic_single','nicwss','clam_sb', 'clam_mb', 'pmil', 'opmil', 'PAMIL'],
                    default='PAMIL',
                    help='type of model (default: clam_sb, clam w/ single attention branch)')

parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--fea_dim', type=int, default=1024,
                     help='the original dimensions of patch embedding')
parser.add_argument('--load_checkpoint', action='store_true', default=False,
                    help='if load the model checkpoint')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'bce'], default='ce',
                     help='slide-level classification loss function (default: ce)')

# dataset configuration -------------------------------------------------------
parser.add_argument('--csv_path', type=str, default=None,
                    help='Optional override for the slide-level metadata CSV.')
parser.add_argument('--data_mag', type=str, default=None,
                    help='Override the magnification suffix appended to slide IDs when reading feature files.')
parser.add_argument('--label_map', nargs='+', default=None,
                    help='Override the default label mapping using KEY=VALUE pairs (e.g. FA=0 PT=1).')
parser.add_argument('--feature_format', choices=['npy', 'h5'], default='npy',
                    help='Storage backend for slide-level feature files.')
parser.add_argument('--h5_feature_key', nargs='+', default=None,
                    help='Dataset keys that contain patch embeddings inside each HDF5 file.')
parser.add_argument('--h5_coord_key', nargs='+', default=None,
                    help='Dataset keys that contain patch coordinates inside each HDF5 file.')
parser.add_argument('--h5_inst_key', nargs='+', default=None,
                    help='Dataset keys that contain instance labels inside each HDF5 file. '
                         "Use 'none' to disable instance label loading.")
parser.add_argument('--h5_file_suffix', type=str, default='',
                    help='Additional string inserted between the slide identifier and the file extension for HDF5 files.')
parser.add_argument('--h5_file_ext', type=str, default='.h5',
                    help='File extension used for HDF5 feature files.')
parser.add_argument('--h5_keep_dtype', action='store_true', default=False,
                    help='Preserve the on-disk dtype instead of converting features to float32.')

### proto attention specific 
parser.add_argument('--num_protos', type=int, default=10, help='the number of protos')
parser.add_argument('--proto_path', type=str, default='./datasets_proto', help='path to pre-clustering prototype')
parser.add_argument('--proto_pred', action='store_true', default=False, help='whether use proto to pred the logits')
parser.add_argument('--proto_weight', type=float, default=0.5, help='the weight of the prototype loss and normal loss')
parser.add_argument('--w_er', type=float, default=0.5, help='the weight of the er loss function')
parser.add_argument('--w_clst', type=float, default=-0.5, 
                    help='the weight of the clst loss function, must be negative')
parser.add_argument('--w_inst', type=float, default=0.5, 
                    help='the weight of the inst loss function')
parser.add_argument('--w_att_er', type=float, default=0.2, 
                    help='the weight of the er loss for attention score')
parser.add_argument('--w_proto_clst', type=float, default=-0.2, 
                    help='the weight of the clst loss to proto -> patch, must be negative')
parser.add_argument('--save_proto', action='store_true', default=False,
                    help='if save the prototype')
parser.add_argument('--inst_pred', action='store_true', default=False, 
                    help='if use the instance loss')
parser.add_argument('--attention_er', action='store_true', default=False, 
                    help='if use the attention similarity loss')
parser.add_argument('--k_sample', type=int, default=8, help='the number to sample')

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024


def _parse_label_map(pairs):
    if not pairs:
        return None
    mapping = {}
    for pair in pairs:
        if '=' not in pair:
            raise ValueError(f"Invalid label_map entry '{pair}'. Expected KEY=VALUE format.")
        key, value = pair.split('=', 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid label_map entry '{pair}'.")
        mapping[key] = int(value)
    return mapping


def _parse_h5_keys(values, *, allow_empty=False):
    if values is None:
        return None
    if allow_empty and len(values) == 1 and values[0].lower() == 'none':
        return tuple()
    return tuple(values)


def _resolve_dataset_class(base_cls):
    if args.feature_format == 'h5':
        if base_cls is Generic_MIL_Dataset:
            return Generic_H5_MIL_Dataset
        raise ValueError('HDF5 loading is not implemented for this dataset configuration.')
    return base_cls


def _resolve_data_mag(default_value, dataset_cls):
    value = args.data_mag if args.data_mag is not None else default_value
    if dataset_cls is Generic_H5_MIL_Dataset and isinstance(value, str):
        if value.strip() == '' or value.lower() == 'none':
            return None
    return value


def _build_dataset(default_csv, default_data_mag, label_dict, *, base_cls=Generic_MIL_Dataset):
    label_override = _parse_label_map(args.label_map)
    dataset_cls = _resolve_dataset_class(base_cls)
    csv_path = args.csv_path if args.csv_path else default_csv
    data_mag = _resolve_data_mag(default_data_mag, dataset_cls)

    dataset_kwargs = dict(
        csv_path=csv_path,
        data_dir=args.data_root_dir,
        data_mag=data_mag,
        shuffle=False,
        seed=10,
        print_info=True,
        label_dict=label_override if label_override is not None else label_dict,
        patient_strat=False,
        ignore=[],
    )

    if dataset_cls is Generic_H5_MIL_Dataset:
        dataset_kwargs.update(
            feature_key=_parse_h5_keys(args.h5_feature_key),
            coord_key=_parse_h5_keys(args.h5_coord_key),
            inst_label_key=_parse_h5_keys(args.h5_inst_key, allow_empty=True),
            file_suffix=args.h5_file_suffix,
            file_ext=args.h5_file_ext,
            use_float32=not args.h5_keep_dtype,
        )

    return dataset_cls(**dataset_kwargs)
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'fea_dim': args.fea_dim,
            'opt': args.opt,
            'num_protos': args.num_protos,
            'proto_path': args.proto_path,
            'proto_pred': args.proto_pred,
            "proto_weight": args.proto_weight,
            'w_er': args.w_er,
            'w_clst': args.w_clst,
            'w_inst': args.w_inst,
            'w_att_er': args.w_att_er}

print('\nLoad Dataset')

dataset = None

if args.task == "gastric_subtype":
    label_dict_ = {'0,0,0': 0, '0,0,1': 1, '0,1,0': 2, '0,1,1': 3,
                   '1,0,0': 4, '1,0,1': 5, '1,1,0': 6, '1,1,1': 7}
    args.n_classes = len(set(label_dict_.values()))

    if args.model_type in ['clam_sb', 'clam_mb', 'PAMIL']:
        dataset = _build_dataset(
            'dataset_csv/gastric_subtyping_npy.csv',
            '1_512',
            label_dict_,
        )

elif args.task == 'gleason_subtype':
    label_dict_ = {'0,0,0': 0, '0,0,1': 1, '0,1,0': 2, '0,1,1': 3,
                   '1,0,0': 4, '1,0,1': 5, '1,1,0': 6, '1,1,1': 7}
    args.n_classes = len(set(label_dict_.values()))

    if args.model_type in ['clam_sb', 'clam_mb', 'PAMIL']:
        dataset = _build_dataset(
            'dataset_csv/gleason_subtyping_npy.csv',
            '0_1024',
            label_dict_,
        )

elif args.task == 'gastric_esd_subtype':
    label_dict_ = {'0,0': 0, '0,1': 1, '1,1': 2}
    args.n_classes = len(set(label_dict_.values()))
    if args.model_type in ['clam_sb', 'clam_mb', 'PAMIL']:
        dataset = _build_dataset(
            'dataset_csv/gastric_esd_subtyping_npy_new.csv',
            '0_512',
            label_dict_,
            base_cls=Generic_MIL_Dataset_gastric_esd,
        )

else:
    raise NotImplementedError

if args.model_type in ['clam_sb', 'clam_mb', 'PAMIL']:
    if dataset is None:
        raise ValueError('Dataset initialisation failed. Check task/model configuration.')
    args.n_classes = dataset.num_classes
    
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)


print('split_dir: ', args.split_dir)
os.makedirs(args.split_dir, exist_ok=True)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()


print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")


