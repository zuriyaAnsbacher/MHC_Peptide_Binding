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
    get_dataset_test, evaluate, evaluate_splitScoreByFreqHLA


def loss_func_BCE(pred, true):
    BCE_Logists = F.binary_cross_entropy_with_logits(pred, true)
    return BCE_Logists


def loss_func_MSE(pred, true):
    MSE = F.mse_loss(pred, true)
    return MSE


def train_epoch(model, train_loader, loss_BCE, loss_MSE, optimizer, var, alpha, device):
    model.train()
    total_loss = 0
    preds1, preds2 = [], []
    trues1, trues2 = [], []
    flags = []
    sigmoid = nn.Sigmoid()

    for batch in train_loader:
        pep, hla, y1, y2, flag, hla_oneHot, emb = batch  # y1 binary, y2 continuous
        pep, hla, y1, y2, flag, hla_oneHot, emb = pep.to(device), hla.to(device), y1.to(device), y2.to(device), \
                                                  flag.to(device), hla_oneHot.to(device), emb.to(device)

        optimizer.zero_grad()

        pred1, pred2 = model(pep, hla, hla_oneHot, emb)
        pred1, pred2 = pred1.view(-1), pred2.view(-1)  # squeeze(1)?

        new_pred1 = torch.multiply(pred1, 1 - flag)
        new_y1 = torch.multiply(y1, 1 - flag)
        new_pred2 = torch.multiply(pred2, flag)
        new_y2 = torch.multiply(y2, flag)

        loss_bce = loss_BCE(new_pred1, new_y1)
        loss_mse = loss_MSE(new_pred2, new_y2) / var

        loss = alpha * loss_bce + (1 - alpha) * loss_mse
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        preds1 += sigmoid(pred1).detach().cpu().numpy().tolist()
        trues1 += y1.tolist()
        preds2 += pred2.tolist()
        trues2 += y2.tolist()
        flags += flag.tolist()

    preds1 = [preds1[i] for i in range(len(flags)) if flags[i] == 0]
    trues1 = [trues1[i] for i in range(len(flags)) if flags[i] == 0]
    preds2 = [preds2[i] for i in range(len(flags)) if flags[i] == 1]
    trues2 = [trues2[i] for i in range(len(flags)) if flags[i] == 1]

    auc = roc_auc_score(trues1, preds1)
    r2 = r2_score(trues2, preds2)
    print("loss train: ", round(total_loss / len(train_loader), 4),
          " | ", "auc train: ", round(auc, 4), " | ", "r2 train: ", round(r2, 4))

    return total_loss / len(train_loader), auc, r2


def run_validation(model, val_loader, loss_BCE, loss_MSE, var, alpha, device):
    model.eval()
    total_loss = 0
    preds1, preds2 = [], []
    trues1, trues2 = [], []
    flags = []
    sigmoid = nn.Sigmoid()

    with torch.no_grad():
        for batch in val_loader:
            pep, hla, y1, y2, flag, hla_oneHot, emb = batch
            pep, hla, y1, y2, flag, hla_oneHot, emb = pep.to(device), hla.to(device), y1.to(device), y2.to(device), \
                                                      flag.to(device), hla_oneHot.to(device), emb.to(device)

            pred1, pred2 = model(pep, hla, hla_oneHot, emb)
            pred1, pred2 = pred1.view(-1), pred2.view(-1)  # squeeze(1)?

            new_pred1 = torch.multiply(pred1, 1 - flag)
            new_y1 = torch.multiply(y1, 1 - flag)
            new_pred2 = torch.multiply(pred2, flag)
            new_y2 = torch.multiply(y2, flag)

            loss_bce = loss_BCE(new_pred1, new_y1)
            loss_mse = loss_MSE(new_pred2, new_y2) / var

            loss = alpha * loss_bce + (1 - alpha) * loss_mse
            total_loss += loss.item()

            preds1 += sigmoid(pred1).detach().cpu().numpy().tolist()
            trues1 += y1.tolist()
            preds2 += pred2.tolist()
            trues2 += y2.tolist()
            flags += flag.tolist()

    preds1 = [preds1[i] for i in range(len(flags)) if flags[i] == 0]
    trues1 = [trues1[i] for i in range(len(flags)) if flags[i] == 0]
    preds2 = [preds2[i] for i in range(len(flags)) if flags[i] == 1]
    trues2 = [trues2[i] for i in range(len(flags)) if flags[i] == 1]

    auc = roc_auc_score(trues1, preds1)
    r2 = r2_score(trues2, preds2)
    print("loss val: ", round(total_loss / len(val_loader), 4),
          "   | ", "auc val: ", round(auc, 4), "   | ", "r2 val: ", round(r2, 4))

    return total_loss / len(val_loader), auc, r2


def run_model(train_loader, val_loader, Pep_model_dict, HLA_model_dict, concat_type, concat_oneHot_dim, concat_emb_dim,
              freeze_weight_AEs, optim_params, model_params, epochs, var, device, early_stopping, is_nni=False):
    pep_model = get_existing_model(Pep_model_dict, model_name='pep')
    hla_model = get_existing_model(HLA_model_dict, model_name='hla')

    # freeze weights of AEs
    if freeze_weight_AEs:
        for param in pep_model.parameters():
            param.requires_grad = False
        for param in hla_model.parameters():
            param.requires_grad = False

    model = HLA_Pep_Model(Pep_model_dict['encoding_dim'], HLA_model_dict['encoding_dim'],
                          pep_model, hla_model, concat_type, concat_oneHot_dim, concat_emb_dim, model_params)
    model.to(device)
    loss_BCE = loss_func_BCE
    loss_MSE = loss_func_MSE
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=optim_params['lr'],  weight_decay=optim_params['weight_decay'])
    alpha = optim_params['alpha']
    train_loss_list = []
    val_loss_list = []
    train_auc_list = []
    val_auc_list = []
    train_r2_list = []
    val_r2_list = []
    min_loss = float('inf')
    max_auc = 0
    counter = 0
    best_model = 'None'

    for epoch in range(1, epochs + 1):
        print(f'Epoch: {epoch} / {epochs}')
        train_loss, train_auc, train_r2 = \
            train_epoch(model, train_loader, loss_BCE, loss_MSE, optimizer, var, alpha, device)
        train_loss_list.append(train_loss)
        train_auc_list.append(train_auc)
        train_r2_list.append(train_r2)
        val_loss, val_auc, val_r2 = \
            run_validation(model, val_loader, loss_BCE, loss_MSE, var, alpha, device)
        val_loss_list.append(val_loss)
        val_auc_list.append(val_auc)
        val_r2_list.append(val_r2)

        if val_loss < min_loss:
            min_loss = val_loss
            counter = 0
            best_model = copy.deepcopy(model)
        elif early_stopping and counter == 30:
            break
        else:
            counter += 1

        if min(train_auc, val_auc) > max_auc:
            max_auc = min(train_auc, val_auc)

        if is_nni:
            nni.report_intermediate_result(max_auc)

    if is_nni:
        nni.report_final_result(max_auc)

    # check if need return epoch - 1
    return best_model, train_loss_list, val_loss_list, train_auc_list, val_auc_list, train_r2_list, val_r2_list, epoch


def main(parameters):
    root = parameters["root"]
    datafile = parameters["datafile"]
    version = parameters["version"]
    HLA_Model_path = parameters["HLA_Model_path"]
    Pep_Model_path = parameters["Pep_Model_path"]
    hla_freq_path = parameters["HLA_Freqs_path"]

    save_path = root + '_CNN_LSTM_FC_Results'
    save_dir = f'{save_path}/CNN_LSTM_FC_version{version}'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    Pep_Header = parameters["Pep_Header"]
    HLA_Seq_Header = parameters["HLA_Seq_Header"]
    HLA_Name_Header = parameters["HLA_Name_Header"]
    Binary_Header = parameters["Binary_Binding_Header"]
    Cont_Header = parameters["Continuous_Binding_Header"]
    Flag_Header = parameters["Flag_Header"]

    cuda_num = parameters["CUDA"]
    epochs = parameters["Epochs"]
    batch_size = parameters["Batch_size"]
    pep_optional_len = parameters["Pep_optional_len"]
    concat_type = parameters["Concat_type"]
    concat_oneHot_dim = parameters["Concat_oneHot_dim"]
    concat_emb_dim = parameters["Concat_emb_dim"]
    split_score_by_freq = parameters["Split_score_by_freq"]
    freeze_weight_AEs = parameters["Freeze_weight_AEs"]

    saved_model_file = parameters["Saved_model_file"]

    if concat_type not in ['oneHot', 'emb', 'None', 'oneHot&zero']:
        print('Error: concat_type has to be "oneHot"/"emb"/"None"/"oneHot&zero" only')
        return

    # best params from NNI
    if concat_type == 'None':
        optim_params = {'lr': 1e-3, 'weight_decay': 7e-7, 'alpha': 0.6}
        model_params = {'hidden_size': 128, 'activ_func': nn.ReLU(), 'dropout': 0.2}
    elif concat_type == 'oneHot':
        optim_params = {'lr': 1e-3, 'weight_decay': 4e-7, 'alpha': 0.6}
        model_params = {'hidden_size': 128, 'activ_func': nn.Tanh(), 'dropout': 0.15}
    elif concat_type == 'emb':
        optim_params = {'lr': 1e-3, 'weight_decay': 7e-7, 'alpha': 0.6}
        model_params = {'hidden_size': 64, 'activ_func': nn.ReLU(), 'dropout': 0.1}
    elif concat_type == 'oneHot&zero':
        optim_params = {'lr': 1e-3, 'weight_decay': 5e-7, 'alpha': 0.65}
        model_params = {'hidden_size': 128, 'activ_func': nn.ReLU(), 'dropout': 0.25}

    HLA_model_dict = load_model_dict(HLA_Model_path)
    Pep_model_dict = load_model_dict(Pep_Model_path)

    print('loading data')
    data = pd.read_csv(datafile)

    x = np.concatenate((data[Pep_Header].values.reshape(-1, 1),
                        data[HLA_Name_Header].values.reshape(-1, 1),
                        data[HLA_Seq_Header].values.reshape(-1, 1)), axis=1)
    y = np.concatenate((data[Binary_Header].values.reshape(-1, 1),
                        data[Cont_Header].values.reshape(-1, 1),
                        data[Flag_Header].values.reshape(-1, 1)), axis=1)

    # hla or pep could be in some groups (train-val-test) but not as a pair
    # stratify by Binary
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y[:, 0])  # I added seed (for comparison)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, stratify=y_train[:, 0]) # I added seed (for comparison)

    headers = [Pep_Header, HLA_Name_Header, HLA_Seq_Header, Binary_Header, Cont_Header, Flag_Header]
    train = pd.DataFrame(np.concatenate((x_train, y_train), axis=1), columns=headers)
    val = pd.DataFrame(np.concatenate((x_val, y_val), axis=1), columns=headers)
    test = pd.DataFrame(np.concatenate((x_test, y_test), axis=1), columns=headers)

    cont_values = train[train[Flag_Header] == 1][Cont_Header]
    var = np.log(cont_values.astype(float), where=cont_values != 0).var()  # variance of log continuous train values
    hla_oneHotMap = get_hla_oneHot_map(hla_freq_path,  n_rows=concat_oneHot_dim)  # 31 most common

    train_data = HLAPepDataset_2Labels(train, headers, hla_oneHotMap, HLA_model_dict["amino_pos_to_num"])
    val_data = HLAPepDataset_2Labels(val, headers, hla_oneHotMap, HLA_model_dict["amino_pos_to_num"])
    test_data = HLAPepDataset_2Labels(test, headers, hla_oneHotMap, HLA_model_dict["amino_pos_to_num"])

    bucket_boundaries = [20 * i for i in pep_optional_len]
    train_sampler = SamplerByLength(train_data, bucket_boundaries, batch_size)
    val_sampler = SamplerByLength(val_data, bucket_boundaries, batch_size)
    test_sampler = SamplerByLength(test_data, bucket_boundaries, batch_size)

    # batch_size = 1 and no shuffle because the batches were organized in sampler
    train_loader = DataLoader(train_data, batch_size=1, batch_sampler=train_sampler, collate_fn=train_data.collate)
    val_loader = DataLoader(val_data, batch_size=1, batch_sampler=val_sampler, collate_fn=val_data.collate)
    test_loader = DataLoader(test_data, batch_size=1, batch_sampler=test_sampler, collate_fn=test_data.collate)

    print('finished loading')

    device = f'cuda:{cuda_num}' if torch.cuda.is_available() else 'cpu'
    model, train_loss_list, val_loss_list, train_auc_list, val_auc_list, train_r2_list, val_r2_list, num_epochs = \
        run_model(train_loader, val_loader, Pep_model_dict, HLA_model_dict, concat_type=concat_type,
                  concat_oneHot_dim=concat_oneHot_dim, concat_emb_dim=concat_emb_dim,
                  freeze_weight_AEs=freeze_weight_AEs, optim_params=optim_params, model_params=model_params,
                  epochs=epochs, var=var, device=device, early_stopping=True)

    plot_loss(train_loss_list, val_loss_list, num_epochs, save_dir)
    plot_auc(train_auc_list, val_auc_list, num_epochs, save_dir)
    plot_r2(train_r2_list, val_r2_list, num_epochs, save_dir)

    if split_score_by_freq:
        evaluate_splitScoreByFreqHLA(model, test_loader, test, test_sampler, batch_size, device)
    else:
        evaluate(model, test_loader, device)

    torch.save({
        'batch_size': batch_size,
        'model_state_dict': model.state_dict()
    }, f'{save_dir}/{saved_model_file}')


def main_nni(stable_params):
    datafile = stable_params["datafile"]
    HLA_Model_path = stable_params["HLA_Model_path"]
    Pep_Model_path = stable_params["Pep_Model_path"]
    hla_freq_path = stable_params["HLA_Freqs_path"]

    Pep_Header = stable_params["Pep_Header"]
    HLA_Seq_Header = stable_params["HLA_Seq_Header"]
    HLA_Name_Header = stable_params["HLA_Name_Header"]
    Binary_Header = stable_params["Binary_Binding_Header"]
    Cont_Header = stable_params["Continuous_Binding_Header"]
    Flag_Header = stable_params["Flag_Header"]

    cuda_num = stable_params["CUDA"]
    epochs = stable_params["Epochs"]
    batch_size = stable_params["Batch_size"]
    pep_optional_len = stable_params["Pep_optional_len"]
    concat_type = stable_params["Concat_type"]
    concat_oneHot_dim = stable_params["Concat_oneHot_dim"]
    freeze_weight_AEs = stable_params["Freeze_weight_AEs"]

    nni_params = nni.get_next_parameter()
    lr = nni_params["lr"]
    weight_decay = nni_params["weight_decay"]
    hidden_size = nni_params["hidden_size"]
    activ_func = nni_params["activ_func"]
    dropout = nni_params["dropout"]

    if concat_type == 'emb':
        concat_emb_dim = nni_params["Concat_emb_dim"]
    else:
        concat_emb_dim = 20

    activ_map = {'relu': nn.ReLU(), 'elu': nn.ELU(), 'tanh': nn.Tanh()}

    optim_params = {'lr': lr, 'weight_decay': weight_decay}
    model_params = {'hidden_size': hidden_size, 'activ_func': activ_map[activ_func], 'dropout': dropout}

    if concat_type not in ['oneHot', 'emb', 'None', 'oneHot&zero']:
        print('Error: concat_type has to be "oneHot"/"emb"/"None"/"oneHot&zero" only')
        return

    HLA_model_dict = load_model_dict(HLA_Model_path)
    Pep_model_dict = load_model_dict(Pep_Model_path)

    print('loading data')
    data = pd.read_csv(datafile)

    x = np.concatenate((data[Pep_Header].values.reshape(-1, 1),
                        data[HLA_Name_Header].values.reshape(-1, 1),
                        data[HLA_Seq_Header].values.reshape(-1, 1)), axis=1)
    y = np.concatenate((data[Binary_Header].values.reshape(-1, 1),
                        data[Cont_Header].values.reshape(-1, 1),
                        data[Flag_Header].values.reshape(-1, 1)), axis=1)

    # hla or pep could be in some groups (train-val-test) but not as a pair
    # stratify by Flag (binary/continuous)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y[:, 2])

    headers = [Pep_Header, HLA_Name_Header, HLA_Seq_Header, Binary_Header, Cont_Header, Flag_Header]
    train = pd.DataFrame(np.concatenate((x_train, y_train), axis=1), columns=headers)
    val = pd.DataFrame(np.concatenate((x_val, y_val), axis=1), columns=headers)

    cont_values = train[train[Flag_Header] == 1][Cont_Header]
    var = np.log(cont_values.astype(float), where=cont_values != 0).var()  # variance of log continuous train values
    hla_oneHotMap = get_hla_oneHot_map(hla_freq_path, n_rows=concat_oneHot_dim)  # 30 most common

    train_data = HLAPepDataset_2Labels(train, headers, hla_oneHotMap, HLA_model_dict["amino_pos_to_num"])
    val_data = HLAPepDataset_2Labels(val, headers, hla_oneHotMap, HLA_model_dict["amino_pos_to_num"])

    bucket_boundaries = [20 * i for i in pep_optional_len]
    train_sampler = SamplerByLength(train_data, bucket_boundaries, batch_size)
    val_sampler = SamplerByLength(val_data, bucket_boundaries, batch_size)

    # batch_size = 1 and no shuffle because the batches were organized in sampler
    train_loader = DataLoader(train_data, batch_size=1, batch_sampler=train_sampler, collate_fn=train_data.collate)
    val_loader = DataLoader(val_data, batch_size=1, batch_sampler=val_sampler, collate_fn=val_data.collate)

    print('finished loading')

    device = f'cuda:{cuda_num}' if torch.cuda.is_available() else 'cpu'
    _, _, _, _, _, _, _, _ = run_model(train_loader, val_loader, Pep_model_dict, HLA_model_dict,
                                       concat_type=concat_type, concat_oneHot_dim=concat_oneHot_dim,
                                       concat_emb_dim=concat_emb_dim, freeze_weight_AEs=freeze_weight_AEs,
                                       optim_params=optim_params, model_params=model_params,
                                       epochs=epochs, var=var, device=device, early_stopping=True,
                                       is_nni=True)


def main_test_on_trained_model(params):
    input_file = f'../Data/IEDB_processed/MHC_and_Pep/mhc_pep01A_701NegNoArtificial_Pep7_11_2labels.csv'

    concat_type = params['Concat_type']
    if concat_type == 'None':
        best_params = {'hidden_size': 128, 'activ_func': nn.ReLU(), 'dropout': 0.2}
    elif concat_type == 'oneHot':
        best_params = {'hidden_size': 128, 'activ_func': nn.Tanh(), 'dropout': 0.15}
    elif concat_type == 'emb':
        best_params = {'hidden_size': 64, 'activ_func': nn.ReLU(), 'dropout': 0.1, 'concat_emb_dim': 25}
    elif concat_type == 'oneHot&zero':
        best_params = {'hidden_size': 128, 'activ_func': nn.ReLU(), 'dropout': 0.25}

    dir_model = f'CNN_LSTM_FC_version8_{concat_type}BestParamsNNI'
    file_model = f'model_best_params_nni_concat{concat_type}.pt'
    combined_model_path = f'HLA_Pep_CNN_LSTM_FC_Results/{dir_model}/{file_model}'

    hla_model_path = params["HLA_Model_path"]
    pep_model_path = params["Pep_Model_path"]
    hla_freq_path = params['HLA_Freqs_path']

    concat_oneHot_dim = params["Concat_oneHot_dim"]
    Pep_Header = params["Pep_Header"]
    HLA_Seq_Header = params["HLA_Seq_Header"]
    HLA_Name_Header = params["HLA_Name_Header"]
    Binary_Header = params["Binary_Binding_Header"]
    Cont_Header = params["Continuous_Binding_Header"]
    Flag_Header = params["Flag_Header"]
    Headers = [Pep_Header, HLA_Name_Header, HLA_Seq_Header, Binary_Header, Cont_Header, Flag_Header]

    device = f'cuda:{params["CUDA"]}' if torch.cuda.is_available() else 'cpu'

    pep_model_dict = load_model_dict(pep_model_path)
    hla_model_dict = load_model_dict(hla_model_path)
    combined_model_dict = load_model_dict(combined_model_path)

    pep_model = get_existing_model(pep_model_dict, model_name='pep')
    hla_model = get_existing_model(hla_model_dict, model_name='hla', is_test=True)

    model = get_combined_model(hla_model, hla_model_dict, pep_model, pep_model_dict,
                               combined_model_dict, best_params, concat_type=concat_type)
    model.to(device)

    df = pd.read_csv(input_file)
    dataset = get_dataset_test(df, Headers, hla_freq_path, hla_model_dict, concat_oneHot_dim)
    test_loader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate, shuffle=False)
    evaluate(model, test_loader, device)


if __name__ == '__main__':
    with open('Parameters/HLA_Pep_parameters.json') as f:
        PARAMETERS = json.load(f)

    if PARAMETERS['IS_TEST']:
        main_test_on_trained_model(PARAMETERS)

    else:
        if not PARAMETERS['IS_NNI']:
            main(PARAMETERS)
        else:
            main_nni(PARAMETERS)



