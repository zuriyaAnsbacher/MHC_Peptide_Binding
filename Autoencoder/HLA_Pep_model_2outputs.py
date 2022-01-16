import os
import json
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score
import matplotlib.pyplot as plt

from AE_with_CNN import CNN_AE
from AE_with_lstm import LSTM_AE
from Loader import HLAPepDataset_2Labels
from Sampler import SamplerByLength


class HLA_Pep_Model(nn.Module):
    def __init__(self, encoding_dim_pep, encoding_dim_hla, pep_AE, hla_AE):
        super().__init__()

        self.encoding_dim_pep = encoding_dim_pep
        self.encoding_dim_hla = encoding_dim_hla
        self.hla_AE = hla_AE
        self.pep_AE = pep_AE

        self.fc1 = nn.Sequential(
            nn.Linear(self.encoding_dim_pep + self.encoding_dim_hla, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.fc2A = nn.Linear(32, 1)
        self.fc2B = nn.Linear(32, 1)

    def forward(self, pep, hla):
        pep_encoded, _ = self.pep_AE(pep)
        hla_encoded, _ = self.hla_AE(hla)
        x = torch.cat((pep_encoded, hla_encoded), dim=1)
        x = self.fc1(x)
        y1 = self.fc2A(x)
        y2 = self.fc2B(x)

        return y1, y2


def get_existing_model(Pep_model_dict, HLA_model_dict, max_len_hla=183):
    # pep_model = LSTM_AE(num_features=20, encoding_dim=Pep_model_dict['encoding_dim'])
    pep_model = LSTM_AE(num_features=20, encoding_dim=Pep_model_dict['enc_dim'])  # todo: change with 'encoding_dim'
    state_dict = Pep_model_dict['model_state_dict']
    pep_model.load_state_dict(state_dict)

    # todo: change to: embedding_dim=HLA_model_dict['embedding_dim']
    hla_model = CNN_AE(max_len=max_len_hla, embedding_dim=15,
                       encoding_dim=HLA_model_dict['encoding_dim'],
                       batch_size=HLA_model_dict['batch_size'])
    state_dict = HLA_model_dict['model_state_dict']
    hla_model.load_state_dict(state_dict)

    return pep_model, hla_model


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
        pep, hla, y1, y2, flag = batch  # y1 binary, y2 continuous
        pep, hla, y1, y2, flag = \
            pep.to(device), hla.to(device), y1.to(device), y2.to(device), flag.to(device)

        optimizer.zero_grad()

        pred1, pred2 = model(pep, hla)
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
            pep, hla, y1, y2, flag = batch
            pep, hla, y1, y2, flag = \
                pep.to(device), hla.to(device), y1.to(device), y2.to(device), flag.to(device)

            pred1, pred2 = model(pep, hla)
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


def run_model(train_loader, val_loader, Pep_model_dict, HLA_model_dict,
              epochs, var, alpha, device, early_stopping):
    pep_model, hla_model = get_existing_model(Pep_model_dict, HLA_model_dict)

    # freeze weights of AEs
    for param in pep_model.parameters():
        param.requires_grad = False
    for param in hla_model.parameters():
        param.requires_grad = False

    # model = HLA_Pep_Model(Pep_model_dict['encoding_dim'], HLA_model_dict['encoding_dim'],
    #                       pep_model, hla_model)
    model = HLA_Pep_Model(Pep_model_dict['enc_dim'], HLA_model_dict['encoding_dim'],  # todo: change to 'encoding_dim'
                          pep_model, hla_model)
    model.to(device)
    loss_BCE = loss_func_BCE
    loss_MSE = loss_func_MSE
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))  # optimizer only on last layers
    train_loss_list = []
    val_loss_list = []
    train_auc_list = []
    val_auc_list = []
    train_r2_list = []
    val_r2_list = []
    min_loss = float('inf')
    counter = 0
    best_model = 'None'

    for epoch in range(1, epochs + 1):
        print(f'Epoch: {epoch} / {epochs}')
        train_loss, train_auc, train_r2 = train_epoch(model, train_loader, loss_BCE, loss_MSE, optimizer, var, alpha, device)
        train_loss_list.append(train_loss)
        train_auc_list.append(train_auc)
        train_r2_list.append(train_r2)
        val_loss, val_auc, val_r2 = run_validation(model, val_loader, loss_BCE, loss_MSE, var, alpha, device)
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

    # check if need return epoch - 1
    return best_model, train_loss_list, val_loss_list, train_auc_list, val_auc_list, train_r2_list, val_r2_list, epoch


def plot_loss(train, val, num_epochs, save_dir):
    epochs = [e for e in range(1, num_epochs + 1)]
    label1 = 'Train'
    label2 = 'Validation'
    plt.figure(2)
    plt.plot(epochs, train, 'bo', color='deepskyblue', label=label1)
    plt.plot(epochs, val, 'b', color='orange', label=label2)
    plt.xlabel('Epoch')
    plt.ylabel('Average loss')
    plt.legend()
    plt.savefig(f'{save_dir}/bce_and_mse_loss.pdf')
    plt.close()


def plot_auc(train_auc, val_auc, num_epochs, save_dir):
    epochs = [e for e in range(1, num_epochs + 1)]
    label1 = 'Train'
    label2 = 'Validation'
    plt.figure(2)
    plt.plot(epochs, train_auc, color='deepskyblue', label=label1)
    plt.plot(epochs, val_auc, color='orange', label=label2)
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.savefig(f'{save_dir}/auc.pdf')
    plt.close()


def plot_r2(train_r2, val_r2, num_epochs, save_dir):
    epochs = [e for e in range(1, num_epochs + 1)]
    label1 = 'Train'
    label2 = 'Validation'
    plt.figure(2)
    plt.plot(epochs, train_r2, color='deepskyblue', label=label1)
    plt.plot(epochs, val_r2, color='orange', label=label2)
    plt.xlabel('Epoch')
    plt.ylabel('R2')
    plt.legend()
    plt.savefig(f'{save_dir}/r2.pdf')
    plt.close()


def evaluate(model, test_loader, device):
    model.eval()
    preds1, preds2 = [], []
    trues1, trues2 = [], []
    flags = []
    sigmoid = nn.Sigmoid()

    with torch.no_grad():
        for batch in test_loader:
            pep, hla, y1, y2, flag = batch
            pep, hla, y1, y2, flag = \
                pep.to(device), hla.to(device), y1.to(device), y2.to(device), flag.to(device)

            pred1, pred2 = model(pep, hla)
            pred1, pred2 = pred1.view(-1), pred2.view(-1)

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
    print("auc test: ", round(auc, 4), " | ", "r2 test: ", round(r2, 4))


def main():
    with open('HLA_Pep_parameters.json') as f:
        parameters = json.load(f)

    root = parameters["root"]
    datafile = parameters["datafile"]
    version = parameters["version"]
    HLA_Model_path = parameters["HLA_Model_path"]
    Pep_Model_path = parameters["Pep_Model_path"]

    save_path = root + '_CNN_LSTM_FC_Results'
    save_dir = f'{save_path}/CNN_LSTM_FC_version{version}'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    Pep_Header = parameters["Pep_Header"]
    HLA_Header = parameters["HLA_Header"]
    Binary_Header = parameters["Binary_Binding_Header"]
    Cont_Header = parameters["Continuous_Binding_Header"]
    Flag_Header = parameters["Flag_Header"]

    epochs = parameters["EPOCHS"]
    batch_size = parameters["BATCH_SIZE"]
    alpha = parameters["Alpha"]
    pep_optional_len = parameters["Pep_optional_len"]
    saved_model_file = parameters["Saved_model_file"]

    # load existing models
    if torch.cuda.is_available():
        HLA_model_dict = torch.load(HLA_Model_path)
        Pep_model_dict = torch.load(Pep_Model_path)
    else:
        HLA_model_dict = torch.load(HLA_Model_path, map_location=torch.device('cpu'))
        Pep_model_dict = torch.load(Pep_Model_path, map_location=torch.device('cpu'))

    print('loading data')
    data = pd.read_csv(datafile)

    x = np.concatenate((data[Pep_Header].values.reshape(-1, 1), data[HLA_Header].values.reshape(-1, 1)), axis=1)
    y = np.concatenate((data[Binary_Header].values.reshape(-1, 1),
                        data[Cont_Header].values.reshape(-1, 1),
                        data[Flag_Header].values.reshape(-1, 1)), axis=1)

    # hla or pep could be in some groups (train-val-test) but not as a pair
    # stratify by Flag (binary/continuous)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y[:, 2])
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, stratify=y_train[:, 2])

    headers = [Pep_Header, HLA_Header, Binary_Header, Cont_Header, Flag_Header]
    train = pd.DataFrame(np.concatenate((x_train, y_train), axis=1), columns=headers)
    val = pd.DataFrame(np.concatenate((x_val, y_val), axis=1), columns=headers)
    test = pd.DataFrame(np.concatenate((x_test, y_test), axis=1), columns=headers)

    var = np.log(
        train[train[Flag_Header] == 1][Cont_Header].astype(float)).var()  # variance of log continuous train values

    train_data = HLAPepDataset_2Labels(train, headers, HLA_model_dict["amino_pos_to_num"])
    val_data = HLAPepDataset_2Labels(val, headers, HLA_model_dict["amino_pos_to_num"])
    test_data = HLAPepDataset_2Labels(test, headers, HLA_model_dict["amino_pos_to_num"])

    bucket_boundaries = [20 * i for i in pep_optional_len]
    train_sampler = SamplerByLength(train_data, bucket_boundaries, batch_size)
    val_sampler = SamplerByLength(val_data, bucket_boundaries, batch_size)
    test_sampler = SamplerByLength(test_data, bucket_boundaries, batch_size)

    # batch_size = 1 and no shuffle because the batches were organized in sampler
    train_loader = DataLoader(train_data, batch_size=1, batch_sampler=train_sampler, collate_fn=train_data.collate)
    val_loader = DataLoader(val_data, batch_size=1, batch_sampler=val_sampler, collate_fn=val_data.collate)
    test_loader = DataLoader(test_data, batch_size=1, batch_sampler=test_sampler, collate_fn=test_data.collate)
    print('finished loading')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, train_loss_list, val_loss_list, train_auc_list, val_auc_list, train_r2_list, val_r2_list, num_epochs = \
        run_model(train_loader, val_loader, Pep_model_dict, HLA_model_dict,
                  epochs=epochs, var=var, alpha=alpha, device=device, early_stopping=True)

    plot_loss(train_loss_list, val_loss_list, num_epochs, save_dir)
    plot_auc(train_auc_list, val_auc_list, num_epochs, save_dir)
    plot_r2(train_r2_list, val_r2_list, num_epochs, save_dir)

    evaluate(model, test_loader, device)

    torch.save({
        'batch_size': batch_size,
        'model_state_dict': model.state_dict()
    }, f'{save_dir}/{saved_model_file}')


if __name__ == '__main__':
    main()
