import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import expit
from sklearn.metrics import roc_auc_score, r2_score

from Models import CNN_AE, LSTM_AE, HLA_Pep_Model
from Loader import HLAPepDataset_2Labels


def load_model_dict(model_path, cuda_num):
    if torch.cuda.is_available():
        model_dict = torch.load(model_path, map_location=f'cuda:{cuda_num}')
    else:
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))
    return model_dict


def get_existing_model(model_dict, model_name, max_len_hla=183, is_test=False):
    if model_name == 'pep':
        model = LSTM_AE(num_features=20, encoding_dim=model_dict['encoding_dim'])
    elif model_name == 'hla':
        model = CNN_AE(max_len=max_len_hla, encoding_dim=model_dict['encoding_dim'],
                       batch_size=1 if is_test else model_dict['batch_size'])
    state_dict = model_dict['model_state_dict']
    model.load_state_dict(state_dict)

    return model


def get_combined_model(hla_model, hla_model_dict, pep_model, pep_model_dict, combined_model_dict,
                       combined_model_params, concat_type='None'):
    combined_model = HLA_Pep_Model(pep_model_dict['encoding_dim'], hla_model_dict['encoding_dim'],
                                   pep_model, hla_model, model_params=combined_model_params, concat_type=concat_type,
                                   concat_oneHot_dim=31, concat_emb_dim=25)
    state_dict = combined_model_dict['model_state_dict']
    combined_model.load_state_dict(state_dict)

    return combined_model


def get_dataset_test(data, headers, hla_freq_path, freq_dict, hla_model_dict, concat_oneHot_dim, concat_type):
    hla_oneHotMap = get_hla_oneHot_map(hla_freq_path, n_rows=concat_oneHot_dim)  # 31 most common
    dataset = HLAPepDataset_2Labels(data, headers, hla_oneHotMap, freq_dict, hla_model_dict["amino_pos_to_num"], concat_type)
    return dataset


def get_hla_oneHot_map(hla_freq_path, n_rows):
    df = pd.read_csv(hla_freq_path, nrows=n_rows)
    HLAs = list(df['HLA'])
    return {HLAs[i]: i for i in range(len(HLAs))}


def get_freq_dict(freq_dir):
    freq_dict = {}
    for filename in os.listdir(freq_dir):
        df = pd.read_csv(os.path.join(freq_dir, filename))
        df['freq'] = np.sqrt(1 / df['freq'])
        freq_dict.update(df.set_index('HLA').to_dict()['freq'])

    return freq_dict
