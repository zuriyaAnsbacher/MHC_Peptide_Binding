import os
import json
import copy
import nni
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score

from Models import HLA_Pep_Model
from Loader import HLAIIPepDataset_2Labels
from Sampler import SamplerByLength
from Plots import plot_loss, plot_auc, plot_r2
from Utils import get_existing_model, load_model_dict, get_combined_model, get_freq_dict


class ModelWrapper:
    def __init__(self, params):
        # ----- configuration -----
        self.version = params.get("version")
        self.cuda_num = params.get("cuda", 0)
        self.device = f'cuda:{self.cuda_num}' if torch.cuda.is_available() else 'cpu'
        self.run_category = params.get("run_category", "classic")  # classic, test, nni

        # ----- paths -----
        self.datafile = params.get("datafile")
        # existing models paths
        self.HLA_model_path = params.get("HLA_model_path")
        self.pep_model_path = params.get("pep_model_path")
        self.combined_model_path = params.get("combined_model_path")
        # results paths
        self.res_dir = params.get("res_dir", "Results")
        self.AE_dir = params.get("AE_dir")
        self.specific_model_dir = params.get("specific_model_dir")
        self.res_path = f'{self.res_dir}/{self.AE_dir}/{self.specific_model_dir}/version{self.version}'
        self.pt_file = params.get("pt_file")
        if self.run_category == "classic":
            self.create_paths()
        # frequencies paths
        self.freq_dir = params.get("freq_dir")
        self.freq_file = params.get("freq_file")

        self.freq_dict = get_freq_dict(self.freq_dir)

        # ----- datafile headers -----
        self.pep_header = params.get("pep_header", "Pep")
        self.HLA_name_header = params.get("HLA_Name_Header", "HLA name")
        self.HLA_seq_header = params.get("HLA_seq_header", "HLA Seq")
        self.binary_header = params.get("binary_binding_header", "Binary binding")
        self.cont_header = params.get("continuous_binding_header", "Continuous binding")
        self.flag_header = params.get("flag_header", "Flag")

        self.pepCore_header = params.get("PepCore_header")
        self.ID_header = params.get("ID_header")

        self.headers = [self.ID_header, self.pepCore_header, self.pep_header, self.HLA_name_header,
                        self.HLA_seq_header, self.binary_header, self.cont_header, self.flag_header]

        # existing string when binding core is unknown
        self.noCore_sign = params.get("noCore_sign", "unknown")

        # ----- flags -----
        self.split_score_by_freq = params.get("split_score_by_freq", False)
        self.loss_weighted_by_freq = params.get("loss_weighted_by_freq", False)
        self.freeze_weight_AEs = params.get("freeze_weight_AEs", False)
        self.transfer_learning = params.get("transfer_learning", False)

        # ----- hyper-parameters -----
        self.epochs = params.get("epochs", 400)
        self.batch_size = params.get("batch_size", 64)
        self.pep_optional_len = params.get("pep_optional_len", [7, 8, 9, 10, 11])
        self.pep_core = params.get("pep_core", 9)
        self.update_core_freq = params.get("update_core_freq", 10)
        self.epoch_to_add_noCore_data = params.get("epoch_to_add_noCore_data", 20)

        self.optim_params, self.architect_params = self.get_params_for_model_and_optimizer()
        if self.run_category == 'nni':
            self.set_nni_params()

        # ----- load AES -----
        self.HLA_model_dict = load_model_dict(self.HLA_model_path, self.cuda_num)
        self.pep_model_dict = load_model_dict(self.pep_model_path, self.cuda_num)

        # ----- for later use -----
        self.train_data, self.val_data, self.test_data = None, None, None
        self.train_loader, self.val_loader, self.test_loader = None, None, None
        self.data_without_core = None
        self.optimizer = None
        self.var = None

    def create_paths(self):
        """
        Create paths for saving results
        """
        dir1 = self.res_dir
        dir2 = os.path.join(self.res_dir, self.AE_dir)
        dir3 = os.path.join(self.res_dir, self.AE_dir, self.specific_model_dir)
        dir4 = self.res_path

        for d in [dir1, dir2, dir3, dir4]:
            if not os.path.exists(d):
                os.mkdir(d)

    def get_params_for_model_and_optimizer(self):
        """
        Best params from NNI
        """
        optim_params = {'lr': 1e-3, 'weight_decay': 7e-7, 'alpha': 0.6}
        if self.transfer_learning:
            optim_params['lr'] = 1e-4
        architect_params = {'hidden_size': 128, 'activ_func': nn.ReLU(), 'dropout': 0.2}

        return optim_params, architect_params

    def set_nni_params(self):
        """
        Adjust to current nni running
        """
        nni_params = nni.get_next_parameter()
        hidden_size = int(nni_params["hidden_size"])
        activ_func = nni_params["activ_func"]
        lr = float(nni_params["lr"])
        weight_decay = float(nni_params["weight_decay"])
        dropout = float(nni_params["dropout"])

        activ_map = {'relu': nn.ReLU(), 'leaky': nn.LeakyReLU(), 'elu': nn.ELU(), 'tanh': nn.Tanh()}
        self.optim_params = {'lr': lr, 'weight_decay': weight_decay, 'alpha': 0.6}
        self.architect_params = {'hidden_size': hidden_size, 'activ_func': activ_map[activ_func], 'dropout': dropout}

    def keep_aside_sample_without_core(self, data):
        """
        When the data contains samples whose cores are unknown, these samples are kept separately.
        Later, after training e model along some epochs, these samples are inserted into the
        trained model to get prediction cores, and merged with the data.
        """
        # in case that all samples have core - do nothing
        if self.noCore_sign not in list(data[self.pepCore_header]):
            return data

        # keep the samples without core aside, to be added to data after first stage of training
        df_no_core = data[data[self.pepCore_header] == self.noCore_sign]
        self.data_without_core = df_no_core

        # data contains now only samples with core
        data = data[data[self.pepCore_header] != self.noCore_sign]
        return data

    @ staticmethod
    def split_Kmer_to_rows(data, Kmer_col_name):
        """
        split the K-mer options to separate rows (so each K-mer will be considered as a sample)
        """
        df_Kmer = data[Kmer_col_name].apply(pd.Series).stack().rename(Kmer_col_name).reset_index()
        data = pd.merge(df_Kmer, data, left_on='level_0', right_index=True, suffixes=(['', '_old']))[data.columns]
        return data

    def createKmers(self, df):
        """
        create all K-mers from peptides
        """
        mer_size = self.pep_core
        Kmer_num = len(df[self.pep_header]) - mer_size + 1
        return [df[self.pep_header][i:i + mer_size] for i in range(Kmer_num)]

    def split_neg_to_Kmer(self, data):
        """
        for each negative (binary) sample, split the original peptide to all optional K-mer
        """
        def createKmers_for_neg(_df):
            # positive binary or continuous (positive/negative) stay with original core (from PepPred)
            if _df[self.binary_header] == 1 or _df[self.flag_header] == 1:
                return _df[self.pepCore_header]
            # negative binary: split to all K-mer
            return self.createKmers(_df)

        data['Kmer'] = data.apply(createKmers_for_neg, axis=1)
        data = self.split_Kmer_to_rows(data, Kmer_col_name='Kmer')

        # replace old core column with Kmer column (that contains the original + splitting negative)
        data.drop([self.pepCore_header], axis=1, inplace=True)
        data.rename(columns={'Kmer': self.pepCore_header}, inplace=True)

        return data

    def data_to_features_and_labels(self, data):
        def reShape(header): return data[header].values.reshape(-1, 1)
        x = np.concatenate((reShape(self.ID_header), reShape(self.pepCore_header), reShape(self.pep_header),
                            reShape(self.HLA_name_header), reShape(self.HLA_seq_header)), axis=1)
        y = np.concatenate((reShape(self.binary_header), reShape(self.cont_header),
                            reShape(self.flag_header)), axis=1)
        return x, y

    def split_data(self, x, y):
        """
        Split data to train-validation-test in classic case, or train-validation in NNI case.
        'stratify' done according binary binding (for proportional ratio).
        Note: hla or peptide could be in some groups (train-val-test) but not as a pair.
         """
        if self.run_category == 'classic':
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y[:, 0])
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, stratify=y_train[:, 0])
        elif self.run_category == 'nni':
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y[:, 0])
            x_test, y_test = None, None
        else:
            raise KeyError("run_category has to be 'classic'/'nni' in split_data")

        return x_train, y_train, x_val, y_val, x_test, y_test

    def np2df(self, x, y):
        return pd.DataFrame(np.concatenate((x, y), axis=1), columns=self.headers)

    def df2dataset(self, df):
        return HLAIIPepDataset_2Labels(df, self.headers, self.freq_dict, self.HLA_model_dict["amino_pos_to_num"])

    def createSampler(self, data):
        bucket_boundaries = [self.pep_model_dict['encoding_dim'] * i for i in self.pep_optional_len]
        return SamplerByLength(data, bucket_boundaries, self.batch_size)

    def createLoader(self, data, data_type, sampler=None):
        is_shuffle = True if data_type != 'test' else False
        if sampler is not None:
            # batch_size=1 and no shuffle because the batches were organized in sampler
            return DataLoader(data, batch_size=1, batch_sampler=sampler, collate_fn=data.collate)
        else:
            return DataLoader(data, batch_size=self.batch_size, shuffle=is_shuffle, collate_fn=data.collate)

    def update_data(self, data, data_type):
        if data_type == 'train':
            self.train_data = data
        elif data_type == 'val':
            self.val_data = data
        elif data_type == 'test':
            self.test_data = data
        else:
            raise KeyError('data_type has to be one of the following: "train"/"val"/"test"')

    def update_loader(self, loader, data_type):
        if data_type == 'train':
            self.train_loader = loader
        elif data_type == 'val':
            self.val_loader = loader
        elif data_type == 'test':
            self.test_loader = loader
        else:
            raise KeyError('data_type has to be one of the following: "train"/"val"')

    def calculate_var(self, train):
        """
        Calculate the variance of log continuous train values.
        Used for continuous loss (divided by var)
        """
        cont_values = train[train[self.flag_header] == 1][self.cont_header]
        self.var = np.log(cont_values.astype(float), where=cont_values != 0).var()

    def create_dataLoaders(self, x_train, y_train, x_val, y_val, x_test=None, y_test=None):
        """
        Create data loaders from np arrays
        """
        descriptions = ['train', 'val']
        all_datasets = [(x_train, y_train), (x_val, y_val)]
        if x_test is not None:
            descriptions.append('test')
            all_datasets.append((x_test, y_test))

        for descrip, xy_data in zip(descriptions, all_datasets):
            x = xy_data[0]
            y = xy_data[1]
            sampler = None

            df = self.np2df(x, y)
            self.update_data(df, descrip)

            if descrip != 'test':
                dataset = self.df2dataset(df)
                # if there is more than one option to pep len, organize batches by len
                if len(self.pep_optional_len) > 1:
                    sampler = self.createSampler(dataset)
                loader = self.createLoader(dataset, descrip, sampler)
                self.update_loader(loader, descrip)

            if descrip == 'train':
                self.calculate_var(df)  # for later use (loss calculation)

    def processing(self):
        print('Loading data ...')

        data = pd.read_csv(self.datafile)
        if self.run_category != 'test':
            data = self.split_neg_to_Kmer(data)
            data = self.keep_aside_sample_without_core(data)
            x, y = self.data_to_features_and_labels(data)
            x_train, y_train, x_val, y_val, x_test, y_test = self.split_data(x, y)
            self.create_dataLoaders(x_train, y_train, x_val, y_val, x_test, y_test)
        else:  # todo: check
            self.test_data = data

        print('Loading finished ... ')

    def loss_BCE(self, pred, true, weights):
        """
        Binary Cross Entropy loss function. Can be used with weights or not
        """
        if self.loss_weighted_by_freq:
            BCE_Logists = F.binary_cross_entropy_with_logits(pred, true, weight=weights)
        else:
            BCE_Logists = F.binary_cross_entropy_with_logits(pred, true)
        return BCE_Logists

    def loss_MSE(self, pred, true, weights):
        """
        Mean Square Error loss function. Can be used with weights or not
        """
        if self.loss_weighted_by_freq:
            MSE = (weights * (pred - true) ** 2).mean()
        else:
            MSE = F.mse_loss(pred, true)
        return MSE

    def predict(self, model, dataloader, group):
        total_loss = 0
        preds1, preds2, trues1, trues2 = [], [], [], []
        flags, freqs, ids = [], [], []
        sigmoid = nn.Sigmoid()
        device = self.device
        alpha = self.optim_params['alpha']

        for batch in dataloader:
            pep, hla, y1, y2, flag, freq2loss, sample_id = batch  # y1 binary, y2 continuous
            pep, hla, y1, y2, flag, freq2loss, sample_id = pep.to(device), hla.to(device), y1.to(device), \
                                                           y2.to(device), flag.to(device), freq2loss.to(device), \
                                                           sample_id.to(device)
            if group == 'train':
                self.optimizer.zero_grad()

            pred1, pred2 = model(pep, hla)
            pred1, pred2 = pred1.view(-1), pred2.view(-1)

            if group != 'test':
                # split to binary and continuous (for losses functions)
                new_pred1 = torch.multiply(pred1, 1 - flag)
                new_y1 = torch.multiply(y1, 1 - flag)
                new_pred2 = torch.multiply(pred2, flag)
                new_y2 = torch.multiply(y2, flag)

                loss_bce = self.loss_BCE(new_pred1, new_y1, weights=freq2loss)
                loss_mse = self.loss_MSE(new_pred2, new_y2, weights=freq2loss) / self.var

                loss = alpha * loss_bce + (1 - alpha) * loss_mse
                total_loss += loss.item()

            if group == 'train':
                loss.backward()
                self.optimizer.step()

            preds1 += sigmoid(pred1).detach().cpu().numpy().tolist()
            trues1 += y1.tolist()
            preds2 += pred2.tolist()
            trues2 += y2.tolist()
            flags += flag.tolist()
            freqs += freq2loss.tolist()
            ids += sample_id.tolist()

        return total_loss, preds1, trues1, preds2, trues2, flags, freqs, ids

    @staticmethod
    def get_scores(preds1, trues1, preds2, trues2, flags):
        """
        Calculate AUC and R2 scores
        """
        # split to binary (preds1, trues1) and continuous (preds2, trues2)
        preds1 = [preds1[i] for i in range(len(flags)) if flags[i] == 0]
        trues1 = [trues1[i] for i in range(len(flags)) if flags[i] == 0]
        preds2 = [preds2[i] for i in range(len(flags)) if flags[i] == 1]
        trues2 = [trues2[i] for i in range(len(flags)) if flags[i] == 1]

        auc = roc_auc_score(trues1, preds1)
        r2 = r2_score(trues2, preds2)

        print(f'AUC: {"%.4f" % round(auc, 4)}  |  R2: {"%.4f" % round(r2, 4)}')
        return auc, r2

    @staticmethod
    def get_scores_by_freq(preds1, trues1, preds2, trues2, flags, freqs):
        """
        Calculate AUC and R2 scores, divided according data frequencies: high, medium and low
        """
        threshold = [(3001, 150000), (101, 3000), (1, 100)]  # todo: check the thresholds
        # freqs are sqrt(1/real_freq), and reverse the order because 1/val
        threshold = [(np.sqrt(1 / y), np.sqrt(1 / x)) for (x, y) in threshold]
        # so threshold = [(1, 0.1), (0.1, 0.22), (0.221, 0)]
        auc_high, r2_high = 0, 0

        for case, thr in zip(['high', 'med ', 'low '], threshold):
            _preds1 = [preds1[i] for i in range(len(flags)) if (flags[i] == 0 and thr[0] < freqs[i] <= thr[1])]
            _trues1 = [trues1[i] for i in range(len(flags)) if (flags[i] == 0 and thr[0] < freqs[i] <= thr[1])]
            _preds2 = [preds2[i] for i in range(len(flags)) if (flags[i] == 1 and thr[0] < freqs[i] <= thr[1])]
            _trues2 = [trues2[i] for i in range(len(flags)) if (flags[i] == 1 and thr[0] < freqs[i] <= thr[1])]

            auc = roc_auc_score(_trues1, _preds1)
            r2 = r2_score(_trues2, _preds2)
            print(f'Case: {case}. AUC: {"%.4f" % round(auc, 4)} | R2: {"%.4f" % round(r2, 4)}')
            if case == 'high':
                auc_high, r2_high = auc, r2

        return auc_high, r2_high

    def train_epoch(self, model):
        model.train()
        print('Train', end='\t\t')
        total_loss, preds1, trues1, preds2, trues2, flags, freqs, _ = self.predict(model, self.train_loader, group='train')
        print(f'Loss: {"%.7f" % round(total_loss / len(self.train_loader), 7)}  |', end='  ')

        if self.split_score_by_freq:
            auc, r2 = self.get_scores_by_freq(preds1, trues1, preds2, trues2, flags, freqs)
        else:
            auc, r2 = self.get_scores(preds1, trues1, preds2, trues2, flags)

        return total_loss / len(self.train_loader), auc, r2

    def validation_epoch(self, model):
        model.eval()
        print('Validation', end='\t')
        with torch.no_grad():
            total_loss, preds1, trues1, preds2, trues2, flags, freqs, _ = self.predict(model, self.val_loader, group='val')
            print(f'Loss: {"%.7f" % round(total_loss / len(self.val_loader), 7)}  |', end='  ')

        if self.split_score_by_freq:
            auc, r2 = self.get_scores_by_freq(preds1, trues1, preds2, trues2, flags, freqs)
        else:
            auc, r2 = self.get_scores(preds1, trues1, preds2, trues2, flags)

        return total_loss / len(self.val_loader), auc, r2

    def evaluate(self, model):
        test_data = self.test_data
        test_data = self.split_to_all_Kmers_and_get_Kmer_with_max_score(test_data, model, is_evaluate=True)

        preds1 = list(test_data['Preds'])
        # divide by 1 because for getting max pred (before), we divide continuous by 1
        preds2 = list(1 / test_data['Preds'])
        trues1 = list(test_data[self.binary_header])
        trues2 = list(test_data[self.cont_header])
        flags = list(test_data[self.flag_header])
        freqs = list(test_data['Freqs'])

        if self.split_score_by_freq:
            _, _ = self.get_scores_by_freq(preds1, trues1, preds2, trues2, flags, freqs)
        else:
            _, _ = self.get_scores(preds1, trues1, preds2, trues2, flags)

    def set_core_with_max_pred(self, df, preds1, preds2, flags, ids, drop_preds):
        """
        for each K-mer group that has been created for each peptide,
        take the one that received max prediction from model
        and update it as the new core in the data
        """
        assert ids == list(df[self.ID_header])  # to ensure that preds are match to their samples

        # create one column with binary and continuous predictions.
        # continuous (preds2) are divided by 1 because lower values mean stronger binding
        df.loc[:, 'Preds'] = [preds1[i] if flags[i] == 0 else 1 / preds2[i] for i in range(len(flags))]
        max_idx = df.groupby([self.ID_header], sort=False)['Preds'].idxmax()
        df = df.iloc[max_idx, :]
        if drop_preds:
            df = df.drop(['Preds'], axis=1)
        return df

    def split_to_all_Kmers_and_get_Kmer_with_max_score(self, data, model,
                                                       update_loader=False, data_type=None, is_evaluate=False):
        """
        For each sample in the data, extract all K-mers from the peptide sequence.
        Afterwards, run the model in "test" mode and get scores for all K-mers.
        Keep in the data the one with the highest score only.
        """
        # split each positive sample to all K-mers
        col_Kmer = pd.DataFrame(data.apply(self.createKmers, axis=1), columns=['Kmer'])  #todo: data['Kmer'] = ...
        data = pd.concat([data, col_Kmer], axis=1)
        data = self.split_Kmer_to_rows(data, Kmer_col_name='Kmer')

        # replace old core column with new (that contains the splitting Kmer)
        data.drop([self.pepCore_header], axis=1, inplace=True)
        data.rename(columns={'Kmer': self.pepCore_header}, inplace=True)

        # create dataloader for running data on the model
        dataset = self.df2dataset(data)
        data_loader = self.createLoader(dataset, data_type='test')
        if update_loader:
            self.update_loader(data_loader, data_type)  # todo: check

        # run model
        if is_evaluate:
            print('\nTest\t\t ', end='\t')
        model.eval()
        with torch.no_grad():
            _, preds1, _, preds2, _, flags, freqs, ids = self.predict(model, data_loader, group='test')

        if is_evaluate:
            data.loc[:, 'Freqs'] = freqs

        drop_preds = False if is_evaluate else True
        data = self.set_core_with_max_pred(data, preds1, preds2, flags, ids, drop_preds)
        return data

    def create_plots(self, train_loss_list, val_loss_list, train_auc_list, val_auc_list,
                     train_r2_list, val_r2_list, epochs_num):
        plot_loss(train_loss_list, val_loss_list, epochs_num, self.res_path)
        plot_auc(train_auc_list, val_auc_list, epochs_num, self.res_path)
        plot_r2(train_r2_list, val_r2_list, epochs_num, self.res_path)

    def run_test(self):
        pep_model = get_existing_model(self.pep_model_dict, model_name='pep')
        hla_model = get_existing_model(self.HLA_model_dict, model_name='hla', is_test=True)
        combined_model_dict = load_model_dict(self.combined_model_path, self.cuda_num)
        model = get_combined_model(hla_model, self.HLA_model_dict, pep_model, self.pep_model_dict,
                                   combined_model_dict, self.architect_params)
        model.to(self.device)
        self.evaluate(model)

    def save_model(self, model):
        torch.save({
            "batch_size": self.batch_size,
            "model_state_dict": model.state_dict()
        }, f'{os.path.join(self.res_path, self.pt_file)}')

    def add_noCore_data(self, model):
        """
        Run the data without core on the model, get the predicted core, and merge it with the original data
        """
        print('Adding data without core ...')
        x_test, y_test = None, None
        new_data = self.data_without_core

        new_data = self.split_to_all_Kmers_and_get_Kmer_with_max_score(new_data, model)

        new_x, new_y = self.data_to_features_and_labels(new_data)
        new_x_train, new_y_train, new_x_val, new_y_val, new_x_test, new_y_test = self.split_data(new_x, new_y)

        x_train, y_train = self.data_to_features_and_labels(self.train_data)
        x_val, y_val = self.data_to_features_and_labels(self.train_data)
        if self.test_data is not None:
            x_test, y_test = self.data_to_features_and_labels(self.test_data)

        x_train = np.concatenate([x_train, new_x_train])
        y_train = np.concatenate([y_train, new_y_train])
        x_val = np.concatenate([x_val, new_x_val])
        y_val = np.concatenate([y_val, new_y_val])
        if self.test_data is not None:
            x_test = np.concatenate([x_test, new_x_test])
            y_test = np.concatenate([y_test, new_y_test])

        self.create_dataLoaders(x_train, y_train, x_val, y_val, x_test, y_test)

    def update_new_cores(self, model):
        def split_pos_neg(_data):
            """
            split data to positive (binary=1 or continuous <=500)
            and negative (binary=0 or continuous > 500) dataframes
            """
            _pos = _data[(_data[self.binary_header] == 1) |
                         ((-1 < _data[self.cont_header]) & (_data[self.cont_header] <= 500))]
            _neg = _data[(_data[self.binary_header] == 0) | (_data[self.cont_header] > 500)]
            return _pos, _neg

        print('Updating new cores ...')
        for data_type in ['train', 'val']:
            data = self.train_data if data_type == 'train' else self.val_data

            # updating done on positive only
            pos, neg = split_pos_neg(data)

            pos = self.split_to_all_Kmers_and_get_Kmer_with_max_score(pos, model, update_loader=True, data_type=data_type)

            # for next epoch: merge negative + updated positive to one dataframe, create loader and save both
            data = pd.concat([pos, neg], axis=0, sort=True)
            data = data.sample(frac=1).reset_index(drop=True)  # shuffle and reset index
            data = data[self.headers]  # reorder columns
            self.update_data(data, data_type)

            dataset = self.df2dataset(data)
            loader = self.createLoader(dataset, data_type)
            self.update_loader(loader, data_type)

    def run_model(self):
        pep_model = get_existing_model(self.pep_model_dict, model_name='pep')
        hla_model = get_existing_model(self.HLA_model_dict, model_name='hla', max_len_hla=266)  # todo: add max to param?

        if self.freeze_weight_AEs:
            for param in list(pep_model.parameters()) + list(hla_model.parameters()):
                param.requires_grad = False

        if self.transfer_learning:
            combined_model_dict = load_model_dict(self.combined_model_path, self.cuda_num)
            model = get_combined_model(hla_model, self.HLA_model_dict, pep_model, self.pep_model_dict,
                                       combined_model_dict, self.architect_params)
        else:
            model = HLA_Pep_Model(self.pep_model_dict['encoding_dim'], self.HLA_model_dict['encoding_dim'],
                                  pep_model, hla_model, self.architect_params)

        model.to(self.device)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=self.optim_params['lr'], weight_decay=self.optim_params['weight_decay'])

        train_loss_list, val_loss_list = [], []
        train_auc_list, val_auc_list = [], []
        train_r2_list, val_r2_list = [], []
        min_loss = float('inf')
        max_auc = 0
        counter = 0
        best_model = 'None'

        for epoch in range(1, self.epochs + 1):
            print(f'--------------------- Epoch: {epoch}/{self.epochs} ----------------------')
            train_loss, train_auc, train_r2 = self.train_epoch(model)
            val_loss, val_auc, val_r2 = self.validation_epoch(model)

            lists = [train_loss_list, train_auc_list, train_r2_list, val_loss_list, val_auc_list, val_r2_list]
            values = [train_loss, train_auc, train_r2, val_loss, val_auc, val_r2]
            for lst, val in zip(lists, values):
                lst.append(val)

            # early stopping, max_counter = 30
            if val_loss < min_loss:
                min_loss = val_loss
                counter = 0
                best_model = copy.deepcopy(model)
            elif counter == 30:
                break
            else:
                counter += 1

            if min(train_auc, val_auc) > max_auc:
                max_auc = min(train_auc, val_auc)

            if self.run_category == 'nni':
                nni.report_intermediate_result(max_auc)

            if epoch == self.epoch_to_add_noCore_data and self.data_without_core is not None:
                self.add_noCore_data(model)

            if epoch % self.update_core_freq == 0:
                self.update_new_cores(model)

        if self.run_category == 'nni':
            nni.report_final_result(max_auc)

        if self.run_category == 'classic':
            self.create_plots(train_loss_list, val_loss_list, train_auc_list, val_auc_list,
                              train_r2_list, val_r2_list, epoch)
        return best_model


if __name__ == '__main__':
    with open('Parameters/HLA-II_Pep_parameters.json') as f:
        PARAMETERS = json.load(f)
    model_wrapper = ModelWrapper(PARAMETERS)
    model_wrapper.processing()

    if model_wrapper.run_category == 'classic':
        BEST_MODEL = model_wrapper.run_model()
        model_wrapper.evaluate(BEST_MODEL)
        model_wrapper.save_model(BEST_MODEL)
    elif model_wrapper.run_category == 'nni':
        _ = model_wrapper.run_model()
    elif model_wrapper.run_category == 'test':
        model_wrapper.run_test()
