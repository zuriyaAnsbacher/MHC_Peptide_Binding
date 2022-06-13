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
from Loader import HLAPepDataset_2Labels
from Sampler import SamplerByLength
from Plots import plot_loss, plot_auc, plot_r2
from Utils import get_existing_model, get_hla_oneHot_map, load_model_dict, get_combined_model, \
    get_dataset_test, get_freq_dict


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

        # ----- datafile headers -----
        self.pep_header = params.get("pep_header", "Pep")
        self.HLA_name_header = params.get("HLA_Name_Header", "HLA name")
        self.HLA_seq_header = params.get("HLA_seq_header", "HLA Seq")
        self.binary_header = params.get("binary_binding_header", "Binary binding")
        self.cont_header = params.get("continuous_binding_header", "Continuous binding")
        self.flag_header = params.get("flag_header", "Flag")
        self.headers = [self.pep_header, self.HLA_name_header, self.HLA_seq_header,
                        self.binary_header, self.cont_header, self.flag_header]

        # ----- concat types -----
        self.concat_type = params.get("concat_type", "None")
        self.concat_oneHot_dim = params.get("concat_oneHot_dim", 31)
        self.concat_emb_dim = params.get("concat_emb_dim", 25)

        # ----- flags -----
        self.split_score_by_freq = params.get("split_score_by_freq", False)
        self.loss_weighted_by_freq = params.get("loss_weighted_by_freq", False)
        self.freeze_weight_AEs = params.get("freeze_weight_AEs", False)
        self.transfer_learning = params.get("transfer_learning", False)

        # ----- hyper-parameters -----
        self.epochs = params.get("epochs", 400)
        self.batch_size = params.get("batch_size", 64)
        self.pep_optional_len = params.get("pep_optional_len", [7, 8, 9, 10, 11])
        self.optim_params, self.architect_params = self.get_params_for_model_and_optimizer()
        if self.run_category == 'nni':
            self.set_nni_params()

        # ----- load AES -----
        self.HLA_model_dict = load_model_dict(self.HLA_model_path, self.cuda_num)
        self.pep_model_dict = load_model_dict(self.pep_model_path, self.cuda_num)

        # ----- for later use -----
        self.train_loader, self.val_loader, self.test_loader = None, None, None
        self.optimizer = None
        self.var = None

    def create_paths(self):
        dir1 = self.res_dir
        dir2 = os.path.join(self.res_dir, self.AE_dir)
        dir3 = os.path.join(self.res_dir, self.AE_dir, self.specific_model_dir)
        dir4 = self.res_path

        for d in [dir1, dir2, dir3, dir4]:
            if not os.path.exists(d):
                os.mkdir(d)

    def get_params_for_model_and_optimizer(self):
        """ Best params from NNI """
        if self.concat_type == 'None':
            optim_params = {'lr': 1e-3, 'weight_decay': 7e-7, 'alpha': 0.6}
            if self.transfer_learning:
                optim_params['lr'] = 1e-4
            architect_params = {'hidden_size': 128, 'activ_func': nn.ReLU(), 'dropout': 0.2}
            # if using Att model: hidden_size: 64, activ_func: nn.Tanh(), att_layers: 9
        elif self.concat_type == 'oneHot':
            optim_params = {'lr': 1e-3, 'weight_decay': 4e-7, 'alpha': 0.6}
            architect_params = {'hidden_size': 128, 'activ_func': nn.Tanh(), 'dropout': 0.15}
        elif self.concat_type == 'emb':
            optim_params = {'lr': 1e-3, 'weight_decay': 7e-7, 'alpha': 0.6}
            architect_params = {'hidden_size': 64, 'activ_func': nn.ReLU(), 'dropout': 0.1}
        elif self.concat_type == 'oneHot&zero':
            optim_params = {'lr': 1e-3, 'weight_decay': 5e-7, 'alpha': 0.65}
            architect_params = {'hidden_size': 128, 'activ_func': nn.ReLU(), 'dropout': 0.25}
        else:
            raise KeyError("concat_type has to be one from the following: 'None'/'oneHot'/'emb'/'oneHot&zero'")

        return optim_params, architect_params

    def set_nni_params(self):
        """ Adjust to current nni running """
        nni_params = nni.get_next_parameter()
        hidden_size = int(nni_params["hidden_size"])
        activ_func = nni_params["activ_func"]
        lr = float(nni_params["lr"])
        weight_decay = float(nni_params["weight_decay"])
        dropout = float(nni_params["dropout"])

        activ_map = {'relu': nn.ReLU(), 'leaky': nn.LeakyReLU(), 'elu': nn.ELU(), 'tanh': nn.Tanh()}
        self.optim_params = {'lr': lr, 'weight_decay': weight_decay, 'alpha': 0.6}
        self.architect_params = {'hidden_size': hidden_size, 'activ_func': activ_map[activ_func], 'dropout': dropout}

    def loss_BCE(self, pred, true, weights):
        """ Binary Cross Entropy loss function. Can be used with weights or not  """
        if self.loss_weighted_by_freq:
            BCE_Logists = F.binary_cross_entropy_with_logits(pred, true, weight=weights)
        else:
            BCE_Logists = F.binary_cross_entropy_with_logits(pred, true)
        return BCE_Logists

    def loss_MSE(self, pred, true, weights):
        """ Mean Square Error loss function. Can be used with weights or not  """
        if self.loss_weighted_by_freq:
            MSE = (weights * (pred - true) ** 2).mean()
        else:
            MSE = F.mse_loss(pred, true)
        return MSE

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

    def calculate_var(self, train):
        """
        Calculate the variance of log continuous train values.
        Used for continuous loss (divided by var)
        """
        cont_values = train[train[self.flag_header] == 1][self.cont_header]
        self.var = np.log(cont_values.astype(float), where=cont_values != 0).var()

    def create_dataLoaders(self, x_train, y_train, x_val, y_val, x_test, y_test):
        """ Create Data Loaders from dataframes """
        headers = self.headers
        bucket_boundaries = [20 * i for i in self.pep_optional_len]  # 20 is encoding_dim of pep on its AE

        hla_oneHotMap = get_hla_oneHot_map(os.path.join(self.freq_dir, self.freq_file),
                                           n_rows=self.concat_oneHot_dim)  # n most common
        freq_dict = get_freq_dict(self.freq_dir)

        def np2df(_x, _y): return pd.DataFrame(np.concatenate((_x, _y), axis=1), columns=headers)

        def df2dataset(_df): return HLAPepDataset_2Labels(_df, headers, hla_oneHotMap, freq_dict,
                                                          self.HLA_model_dict["amino_pos_to_num"], self.concat_type)

        def createSampler(_data): return SamplerByLength(_data, bucket_boundaries, self.batch_size)  #todo: too long. try to make it more efficient

        # batch_size = 1 and no shuffle because the batches were organized in sampler
        def createLoader(_data, _sampler): return DataLoader(_data, batch_size=1, batch_sampler=_sampler,
                                                             collate_fn=_data.collate)

        train = np2df(x_train, y_train)
        val = np2df(x_val, y_val)

        self.calculate_var(train)

        train_data = df2dataset(train)
        val_data = df2dataset(val)

        train_sampler = createSampler(train_data)
        val_sampler = createSampler(val_data)

        self.train_loader = createLoader(train_data, train_sampler)
        self.val_loader = createLoader(val_data, val_sampler)

        if x_test is not None:
            test = np2df(x_test, y_test)
            test_data = df2dataset(test)
            test_sampler = createSampler(test_data)
            self.test_loader = createLoader(test_data, test_sampler)

    def processing(self):
        print('Loading data ...')
        data = pd.read_csv(self.datafile)

        if self.run_category != 'test':
            def reShape(header): return data[header].values.reshape(-1, 1)

            x = np.concatenate((reShape(self.pep_header), reShape(self.HLA_name_header),
                                reShape(self.HLA_seq_header)), axis=1)
            y = np.concatenate((reShape(self.binary_header), reShape(self.cont_header),
                                reShape(self.flag_header)), axis=1)

            x_train, y_train, x_val, y_val, x_test, y_test = self.split_data(x, y)
            self.create_dataLoaders(x_train, y_train, x_val, y_val, x_test, y_test)
        else:
            freq_dict = get_freq_dict(self.freq_dir)
            dataset = get_dataset_test(data, self.headers, os.path.join(self.freq_dir, self.freq_file),
                                       freq_dict, self.HLA_model_dict, self.concat_oneHot_dim, self.concat_type)
            self.test_loader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate, shuffle=False)

        print('Loading finished ... ')

    def get_dataloader(self, group):
        if group == 'train':
            return self.train_loader
        elif group == 'val':
            return self.val_loader
        elif group == 'test':
            return self.test_loader
        else:
            raise KeyError("'group' has to be one of the following: 'train'/'val'/'test'")

    def predict(self, model, group):
        total_loss = 0
        preds1, preds2 = [], []
        trues1, trues2 = [], []
        flags = []
        freqs = []
        sigmoid = nn.Sigmoid()
        device = self.device
        alpha = self.optim_params['alpha']
        dataloader = self.get_dataloader(group)

        for batch in dataloader:
            pep, hla, y1, y2, flag, freq2loss, hla_oneHot, emb = batch  # y1 binary, y2 continuous
            pep, hla, y1, y2, flag, freq2loss, hla_oneHot, emb = pep.to(device), hla.to(device), y1.to(device), \
                                                                 y2.to(device), flag.to(device), freq2loss.to(device), \
                                                                 hla_oneHot.to(device), emb.to(device)
            if group == 'train':
                self.optimizer.zero_grad()

            pred1, pred2 = model(pep, hla, hla_oneHot, emb)
            pred1, pred2 = pred1.view(-1), pred2.view(-1)

            if group != 'test':
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

        return total_loss, preds1, trues1, preds2, trues2, flags, freqs

    @staticmethod
    def get_scores(preds1, trues1, preds2, trues2, flags):
        preds1 = [preds1[i] for i in range(len(flags)) if flags[i] == 0]
        trues1 = [trues1[i] for i in range(len(flags)) if flags[i] == 0]
        preds2 = [preds2[i] for i in range(len(flags)) if flags[i] == 1]
        trues2 = [trues2[i] for i in range(len(flags)) if flags[i] == 1]

        auc = roc_auc_score(trues1, preds1)
        r2 = r2_score(trues2, preds2)

        print(f'AUC: {"%.4f" % round(auc, 4)} | R2: {"%.4f" % round(r2, 4)}')
        return auc, r2

    @staticmethod
    def get_scores_by_freq(preds1, trues1, preds2, trues2, flags, freqs):
        threshold = [(3001, 150000), (101, 3000), (1, 100)]  # todo: check the thresholds!
        # freqs are sqrt(1/real_freq), and reverse the order because 1/val
        threshold = [(np.sqrt(1 / y), np.sqrt(1 / x)) for (x, y) in threshold]
        # threshold = [(1, 0.1), (0.1, 0.22), (0.221, 0)]
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
        print('Train')
        total_loss, preds1, trues1, preds2, trues2, flags, freqs = self.predict(model, group='train')
        print(f'loss: {round(total_loss / len(self.train_loader), 7)}')

        if self.split_score_by_freq:
            auc, r2 = self.get_scores_by_freq(preds1, trues1, preds2, trues2, flags, freqs)
        else:
            auc, r2 = self.get_scores(preds1, trues1, preds2, trues2, flags)

        return total_loss / len(self.train_loader), auc, r2

    def validation_epoch(self, model):
        model.eval()
        print('Validation')
        with torch.no_grad():
            total_loss, preds1, trues1, preds2, trues2, flags, freqs = self.predict(model, group='val')
            print(f'loss: {round(total_loss / len(self.val_loader), 7)}')

        if self.split_score_by_freq:
            auc, r2 = self.get_scores_by_freq(preds1, trues1, preds2, trues2, flags, freqs)
        else:
            auc, r2 = self.get_scores(preds1, trues1, preds2, trues2, flags)

        return total_loss / len(self.val_loader), auc, r2

    def evaluate(self, model):
        model.eval()
        print('Test')
        with torch.no_grad():
            _, preds1, trues1, preds2, trues2, flags, freqs = self.predict(model, group='test')

        if self.split_score_by_freq:
            _, _ = self.get_scores_by_freq(preds1, trues1, preds2, trues2, flags, freqs)
        else:
            _, _ = self.get_scores(preds1, trues1, preds2, trues2, flags)

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
                                   combined_model_dict, self.architect_params, self.concat_type)
        model.to(self.device)
        self.evaluate(model)

    def save_model(self, model):
        torch.save({  # todo: add more fields?
            "batch_size": self.batch_size,
            "model_state_dict": model.state_dict()
        }, f'{os.path.join(self.res_path, self.pt_file)}')

    def run_model(self):
        pep_model = get_existing_model(self.pep_model_dict, model_name='pep')
        hla_model = get_existing_model(self.HLA_model_dict, model_name='hla')

        if self.freeze_weight_AEs:
            for param in list(pep_model.parameters()) + list(hla_model.parameters()):
                param.requires_grad = False

        if self.transfer_learning:
            combined_model_dict = load_model_dict(self.combined_model_path, self.cuda_num)
            model = get_combined_model(hla_model, self.HLA_model_dict, pep_model, self.pep_model_dict,
                                       combined_model_dict, self.architect_params, self.concat_type)
        else:
            model = HLA_Pep_Model(self.pep_model_dict['encoding_dim'], self.HLA_model_dict['encoding_dim'],
                                  pep_model, hla_model, self.architect_params, self.concat_type,
                                  self.concat_oneHot_dim, self.concat_emb_dim)

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
            print(f'Epoch: {epoch} / {self.epochs}')
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

        if self.run_category == 'nni':
            nni.report_final_result(max_auc)

        if self.run_category == 'classic':
            # check if need epoch - 1
            self.create_plots(train_loss_list, val_loss_list, train_auc_list, val_auc_list,
                              train_r2_list, val_r2_list, epoch)
        return best_model


if __name__ == '__main__':
    with open('Parameters/HLA_Pep_parameters.json') as f:
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
