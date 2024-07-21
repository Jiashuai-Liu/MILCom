from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats

from torch.utils.data import Dataset
import h5py
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
from utils.utils import generate_split, nth, make_weights_for_balanced_classes_split

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
    splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
    # print(f'splits is {splits}')  # for test
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index = True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

    df.to_csv(filename)
    print()

class Generic_WSI_Classification_Dataset(Dataset):
    def __init__(self,
        csv_path = 'dataset_csv/ccrcc_clean.csv',
        shuffle = False, 
        seed = 7, 
        print_info = True,
        label_dict = {},
        filter_dict = {},
        ignore=[],
        patient_strat=False,
        label_col = None,
        patient_voting = 'max'
        ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.label_dict = label_dict
        self.num_classes = len(set(self.label_dict.values()))
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = None
        if not label_col:
            label_col = 'label'
        self.label_col = label_col

        slide_data = pd.read_csv(csv_path, dtype={'slide_id':str, 'case_id':str})
        slide_data = self.filter_df(slide_data, filter_dict)
        slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)

        ###shuffle data
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data.values)

        self.slide_data = slide_data

        self.patient_data_prep(patient_voting)
        self.cls_ids_prep()

        if print_info:
            self.summarize()

    def cls_ids_prep(self):
        # store ids corresponding each class at the patient or case level
        self.patient_cls_ids = [[] for i in range(self.num_classes)]		
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        # store ids corresponding each class at the slide level
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def patient_data_prep(self, patient_voting='max'):
        patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
        patient_labels = []

        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations].values
            if patient_voting == 'max':
                label = label.max() # get patient label (MIL convention)
            elif patient_voting == 'maj':
                label = stats.mode(label)[0]
            else:
                raise NotImplementedError
            patient_labels.append(label)

        self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}

    @staticmethod
    def df_prep(data, label_dict, ignore, label_col):
        if label_col != 'label':
            data['label'] = data[label_col].copy()
        
        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        for i in data.index:
            key = data.loc[i, 'label']
            data.at[i, 'label'] = label_dict[key]

        return data

    def filter_df(self, df, filter_dict={}):
        if len(filter_dict) > 0:
            filter_mask = np.full(len(df), True, bool)
            # assert 'label' not in filter_dict.keys()
            for key, val in filter_dict.items():
                mask = df[key].isin(val)
                filter_mask = np.logical_and(filter_mask, mask)
            df = df[filter_mask]
        return df

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])

        else:
            return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

    def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
        settings = {
                    'n_splits' : k, 
                    'val_num' : val_num, 
                    'test_num': test_num,
                    'label_frac': label_frac,
                    'seed': self.seed,
                    'custom_test_ids': custom_test_ids
                    }

        if self.patient_strat:
            settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
        else:
            settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

        self.split_gen = generate_split(**settings)

    def set_splits(self,start_from=None):
        if start_from:
            ids = nth(self.split_gen, start_from)

        else:
            ids = next(self.split_gen)

        if self.patient_strat:
            slide_ids = [[] for i in range(len(ids))] 

            for split in range(len(ids)): 
                for idx in ids[split]:
                    case_id = self.patient_data['case_id'][idx]
                    slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
                    slide_ids[split].extend(slide_indices)

            self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

        else:
            self.train_ids, self.val_ids, self.test_ids = ids

    def get_split_from_df(self, all_splits, split_key='train'):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, data_dir=self.data_dir, data_mag=self.data_mag, 
                                  task=self.task, num_classes=self.num_classes)
        else:
            split = None

        return split

    def get_merged_split_from_df(self, all_splits, split_keys=['train']):
        merged_split = []
        for split_key in split_keys:
            split = all_splits[split_key]
            split = split.dropna().reset_index(drop=True).tolist()
            merged_split.extend(split)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(merged_split)
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
        else:
            split = None

        return split


    def return_splits(self, from_id=True, csv_path=None):


        if from_id:
            if len(self.train_ids) > 0:
                train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
                train_split = Generic_Split(train_data, data_dir=self.data_dir, data_mag=self.data_mag, 
                                            task=self.task, num_classes=self.num_classes)

            else:
                train_split = None

            if len(self.val_ids) > 0:
                val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
                val_split = Generic_Split(val_data, data_dir=self.data_dir, data_mag=self.data_mag, 
                                          task=self.task, num_classes=self.num_classes)

            else:
                val_split = None

            if len(self.test_ids) > 0:
                test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
                test_split = Generic_Split(test_data, data_dir=self.data_dir, data_mag=self.data_mag, 
                                           task=self.task, num_classes=self.num_classes)

            else:
                test_split = None


        else:
            assert csv_path 
            all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)

            train_split = self.get_split_from_df(all_splits, 'train')
            val_split = self.get_split_from_df(all_splits, 'val')
            test_split = self.get_split_from_df(all_splits, 'test')

        return train_split, val_split, test_split

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

    def test_split_gen(self, return_descriptor=False):

        if return_descriptor:
            index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
            columns = ['train', 'val', 'test']
            df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
                            columns= columns)

        count = len(self.train_ids)
        print('\nnumber of training samples: {}'.format(count))
        labels = self.getlabel(self.train_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'train'] = counts[u]

        count = len(self.val_ids)
        print('\nnumber of val samples: {}'.format(count))
        labels = self.getlabel(self.val_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'val'] = counts[u]

        count = len(self.test_ids)
        print('\nnumber of test samples: {}'.format(count))
        labels = self.getlabel(self.test_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'test'] = counts[u]

        assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
        assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
        assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

        if return_descriptor:
            return df

    def save_split(self, filename):
        train_split = self.get_list(self.train_ids)
        val_split = self.get_list(self.val_ids)
        test_split = self.get_list(self.test_ids)
        df_tr = pd.DataFrame({'train': train_split})
        df_v = pd.DataFrame({'val': val_split})
        df_t = pd.DataFrame({'test': test_split})
        df = pd.concat([df_tr, df_v, df_t], axis=1) 
        df.to_csv(filename, index = False)


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
    def __init__(self, data_dir, data_mag, task, **kwargs):

        super(Generic_MIL_Dataset, self).__init__(**kwargs)
        
        self.data_dir = data_dir
        self.data_mag = data_mag
        self.task = task
    
    def translabel(self, label, task):
        
        
        if task == 'gastric_esd_subtype':
            label_dict = [[0,0],[0,1],[1,1]] # gastric_esd
#             label_dict = [[0,0,1],[0,1,0],[0,1,1],[1,0,1],[1,1,1]] # gastric_esd tumor normal_gland inflammation 
        elif task == 'gastric_subtype':
            label_dict = [[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,1,0],[1,0,1],[1,1,1]] # gastric low high muc
        elif task == 'gleason_subtype':
            label_dict = [[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,1,0], [1,1,1]] # gleason, delete '1,0,1':5,
        elif task == 'crc_subtype':
            label_dict =  [[1, 1, 1, 0, 0], [1, 1, 0, 1, 0], [1, 1, 0, 0, 1], [1, 0, 1, 1, 0], [1, 0, 1, 0, 1], [1, 0, 0, 1, 1], [0, 1, 1, 1, 0], [0, 1, 1, 0, 1], [0, 1, 0, 1, 1], [0, 0, 1, 1, 1]] # crc
        else:
            raise NotImplementedError
            
        return label_dict[label]
    
    # def trans_inst_label_Gastric(self, inst_label):
    #     inst_label[inst_label==0]=4  # for normal
    #     inst_label[inst_label==1]=0  # for pap
    #     inst_label[inst_label==2]=1  # for tub1
    #     inst_label[inst_label==3]=1  # for tub2
    #     inst_label[inst_label==4]=2  # for por1
    #     inst_label[inst_label==5]=2  # for por2
    #     inst_label[inst_label==6]=5  # for sig->other
    #     inst_label[inst_label==-1]=5  # for background, not annotated
    #     inst_label[inst_label==7]=3  # for muc
    #     return inst_label

    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'][idx]
        label = self.slide_data['label'][idx]
        label = self.translabel(label, self.task)
        data_dir = self.data_dir

        full_path = os.path.join(data_dir,'{}_{}.npy'.format(slide_id, self.data_mag))
        
        record = np.load(full_path, allow_pickle=True)
        
        features = record[()]['feature2']
        # features = features.reshape(features.shape[0]*features.shape[1], features.shape[-1])
        
        coords = record[()]['index']

        inst_label = record[()]['inst_label']
        inst_label = np.array(inst_label)
        if self.task=='gastric_esd_subtype':
            inst_label[inst_label==1]=3
            inst_label[inst_label==2]=1
            inst_label[inst_label==3]=2

        return features, label, coords, inst_label


class Generic_Split(Generic_MIL_Dataset): 
    def __init__(self, slide_data, data_dir=None, data_mag=None, task=None, num_classes=2):
        
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.task = task
        self.num_classes = num_classes
        self.data_mag = data_mag
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def __len__(self):
        return len(self.slide_data)

def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    label = torch.LongTensor([item[1] for item in batch])
    coords = [item[2] for item in batch]
    inst_label = [item[3] for item in batch]
    return [img, label, coords, inst_label]

def get_split_loader(split_dataset, training = False, testing = False, weighted = False):
    """
		return either the validation loader or training loader 
	"""
    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    if testing:
        ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
        loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate_MIL, **kwargs )
    elif training:
        if weighted:
            weights = make_weights_for_balanced_classes_split(split_dataset)
            loader = DataLoader(split_dataset, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_MIL, **kwargs)	
        else:
            loader = DataLoader(split_dataset, batch_size=1, sampler = RandomSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
    else:
        loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate_MIL, **kwargs)

    return loader