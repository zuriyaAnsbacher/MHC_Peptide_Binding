import os
import csv
import json
import nni
import torch
import torch.nn.functional as F
import pandas as pd
from random import shuffle
from sklearn.model_selection import train_test_split
import torch.optim as optim
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import copy
from numpy import array
from collections import defaultdict

from Models import LSTM_AE


def loss_func(x, y):
    MSE = F.mse_loss(x, y, reduction='sum')
    return MSE


def load_all_data(datafile, seq):
    df = pd.read_csv(datafile)
    data = list(df[seq])
    random.shuffle(data)
    data = list(set(data))  # maybe unnecessary, because I removed duplicates in data processing
    return data


def find_max_len(sequences):
    return max([len(seq) for seq in sequences])


def data_preprocessing_lstm(string_set, amino_to_ix):
    one_hot = {a: [0] * 20 for a in amino_to_ix.keys()}
    for key in one_hot:
        one_hot[key][amino_to_ix[key]] = 1
    one_vecs = []
    for seq in string_set:
        v = []
        for h in seq:
            v += one_hot[h]
        one_vecs.append(v)
    return one_vecs


def lengths_clones(samples):
    d = defaultdict(list)
    for s in samples:
        d[len(s)].append(s)
    result = []
    # for n in sorted(d, reverse=True):
    for n in sorted(d.keys(), reverse=True):
        clone = d[n]
        result.append(
            torch.Tensor(array(clone).reshape(len(clone), len(clone[0]))))
    return result


def convert2tensor(samples):
    samples = torch.Tensor(array(samples).reshape(len(samples), len(samples[0])))
    return samples


def split_to_batches(samples, batch_size):
    batches = list()
    for i in range(len(samples)):
        batch = torch.split(samples[i], batch_size)
        for j in range(len(batch)):
            batches.append(batch[j])
    return batches


def split_to_batches_1clone(samples, batch_size):
    batches = list(torch.split(samples, batch_size))
    return batches


def train_epoch(batches, model, loss_function, optimizer, device):
    model.train()
    shuffle(batches)
    total_loss = 0
    for i in tqdm(range(len(batches))):
        padded_seqs = batches[i]
        # Move to GPU
        padded_seqs = padded_seqs.to(device)
        model.zero_grad()
        mu, pred = model(padded_seqs)
        # Compute loss
        loss = loss_function(pred, padded_seqs)
        # Update model weights
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # Return average loss
    print("loss:", total_loss / len(batches))
    return total_loss / len(batches)


def train_model(batches, validation_batches, optim_params, model_params, encoding_dim, epochs, device,
                early_stopping=False, is_nni=False):
    model = LSTM_AE(num_features=20, encoding_dim=encoding_dim, model_params=model_params)
    model.to(device)
    loss_function = loss_func
    optimizer = optim.Adam(model.parameters(), lr=optim_params['lr'], weight_decay=optim_params['weight_decay'])
    train_loss_list = list()
    val_loss_list = list()
    min_loss = float('inf')
    counter = 0
    best_model = 'None'

    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1} / {epochs}')
        train_loss = train_epoch(batches, model, loss_function, optimizer, device)
        train_loss_list.append(train_loss)
        val_loss = run_validation(model, validation_batches, loss_function, device)
        val_loss_list.append(val_loss)
        print("Val loss:", val_loss)

        if val_loss < min_loss:
            min_loss = val_loss
            counter = 0
            best_model = copy.deepcopy(model)
        elif early_stopping and counter == 80:
            break
        else:
            counter += 1

        if is_nni:
            nni.report_intermediate_result(val_loss)

    if is_nni:
        nni.report_final_result(min_loss)

    num_epochs = epoch + 1
    if best_model == 'None':
        return model, train_loss_list, val_loss_list, num_epochs
    else:
        return best_model, train_loss_list, val_loss_list, num_epochs


def read_pred(pred, ix_to_amino):
    batch_SEQs = []
    for seq in pred:
        c_index = torch.argmax(seq, dim=1)
        s = ''
        for index in c_index:
            s += ix_to_amino[index.item()]
        batch_SEQs.append(s)
    return batch_SEQs


def count_mistakes(true_SEQs, pred_SEQs):
    mis = 0
    for i in range(min(len(true_SEQs), len(pred_SEQs))):
        if not true_SEQs[i] == pred_SEQs[i]:
            mis += 1
    return mis


def evaluate(save_path, batches, model, ix_to_amino, max_len, encoding_dim, device, EPOCHS, set_string):
    model.eval()
    print(f'{set_string} evaluating')
    shuffle(batches)
    acc = 0
    acc_1mis = 0
    acc_2mis = 0
    acc_3mis = 0
    count = 0
    for i in tqdm(range(len(batches))):
        with torch.no_grad():
            padded_SEQs = batches[i]
            true_SEQs = read_pred(padded_SEQs.unsqueeze(1).view(padded_SEQs.size(0), -1, 20), ix_to_amino)
            # Move to GPU
            padded_SEQs = padded_SEQs.to(device)
            # model.zero_grad()
            mu, pred = model(padded_SEQs)
            pred = pred.unsqueeze(1)
            pred_SEQs = read_pred(pred.view(pred.size(0), -1, 20), ix_to_amino)
            for j in range(len(batches[i])):
                count += 1
                if len(true_SEQs[j]) != len(pred_SEQs[j]):
                    continue
                mis = count_mistakes(true_SEQs[j], pred_SEQs[j])
                if mis == 0:
                    acc += 1
                if mis <= 1:
                    acc_1mis += 1
                if mis <= 2:
                    acc_2mis += 1
                if mis <= 3:
                    acc_3mis += 1
    acc /= count
    acc_1mis /= count
    acc_2mis /= count
    acc_3mis /= count
    print("acc:", acc)
    print("acc_1mis:", acc_1mis)
    print("acc_2mis:", acc_2mis)
    print("acc_3mis:", acc_3mis)

    with open(f'{save_path}/lstm_model_{set_string}_accuracy_encoding_dim={encoding_dim}.csv', 'w') as csvfile:
        fieldnames = ['Accuracy', '1 Mismatch', '2 Mismatches', '3 Mismatches']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Accuracy': str(acc), '1 Mismatch': str(acc_1mis),
                         '2 Mismatches': str(acc_2mis), '3 Mismatches': str(acc_3mis)})


def run_validation(model, validation, loss_function, device):
    model.eval()
    shuffle(validation)
    total_loss_val = 0
    for i in range(len(validation)):
        with torch.no_grad():
            padded_tcrs = validation[i]
            # Move to GPU
            padded_tcrs = padded_tcrs.to(device)
            mu, pred = model(padded_tcrs)
            # Compute loss
            loss = loss_function(pred, padded_tcrs)
            total_loss_val += loss.item()
    return total_loss_val / len(validation)


def plot_loss(train, val, encoding_dim, num_epochs, save_dir, real_epochs):
    epochs = [e for e in range(num_epochs)]
    label1 = 'Train'
    label2 = 'Validation'
    plt.figure(2)
    plt.plot(epochs, train, 'bo', color='mediumaquamarine', label=label1)
    plt.plot(epochs, val, 'b', color='cornflowerblue', label=label2)
    plt.xlabel('Epoch')
    plt.ylabel('Average loss')
    plt.legend()
    plt.savefig(f'{save_dir}/lstm_model_loss_function_encoding_dim={encoding_dim}.pdf', bbox_inches="tight",
                pad_inches=1)
    plt.close()


def main(parameters):
    root = parameters["root"]  # path to data set
    datafile = parameters['datafile']
    SEQ = parameters["SEQ"]
    HLA_or_Pep = parameters["HLA_or_Pep"]
    epochs = parameters["EPOCHS"]  # number of epochs for each model
    encoding_dim = parameters["ENCODING_DIM"]  # number of dimensions in the embedded space
    batch_size = parameters["BATCH_SIZE"]
    version = parameters["version"]

    optim_params = {'lr': 1e-3, 'weight_decay': 0}
    model_params = {'num_layers_enc': 1, 'num_layers_dec': 1, 'dropout_enc': 0, 'dropout_dec': 0}

    save_path = root + '_LSTM_Results'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_dir = f'{save_path}/LSTM_version{version}'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    amino_to_ix = {amino: index for index, amino in enumerate(amino_acids)}
    ix_to_amino = {index: amino for index, amino in enumerate(amino_acids)}

    print('loading data')
    sequences = load_all_data(datafile, SEQ)
    print('finished loading')

    # tcrs = list(set([t[0][:-1] for t in tcrs]))
    vecs_data = data_preprocessing_lstm(sequences, amino_to_ix)

    # train + test sets
    train, test, _, _ = train_test_split(vecs_data, vecs_data, test_size=0.2)
    train, validation, _, _ = train_test_split(train, train, test_size=0.15)

    if HLA_or_Pep == 'HLA':  # all sequences have same len
        train_clones = convert2tensor(train)
        test_clones = convert2tensor(test)
        validation_clones = convert2tensor(validation)

        train_batches = split_to_batches_1clone(train_clones, batch_size)
        test_batches = split_to_batches_1clone(test_clones, batch_size)
        validation_batches = split_to_batches_1clone(validation_clones, batch_size)

    elif HLA_or_Pep == 'Pep':  # variable len
        train_clones = lengths_clones(train)
        test_clones = lengths_clones(test)
        validation_clones = lengths_clones(validation)

        train_batches = split_to_batches(train_clones, batch_size)
        test_batches = split_to_batches(test_clones, batch_size)
        validation_batches = split_to_batches(validation_clones, batch_size)

    else:
        print("Error: Sequences type has to be 'HLA' or 'Pep'")
        return

    max_len = find_max_len([seq for seq in sequences])
    print('max_len:', max_len)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, train_loss_list, val_loss_list, num_epochs = train_model(
        train_batches, validation_batches, optim_params, model_params,
        encoding_dim=encoding_dim, epochs=epochs,
        device=device, early_stopping=True)
    plot_loss(train_loss_list, val_loss_list, encoding_dim, num_epochs, save_dir, epochs)
    evaluate(save_dir, train_batches, model, ix_to_amino, max_len, encoding_dim, device, epochs, 'Train')
    evaluate(save_dir, test_batches, model, ix_to_amino, max_len, encoding_dim, device, epochs, 'Test')

    # add embedding dim to torch.save (?)
    torch.save({
        'amino_to_ix': amino_to_ix,
        'ix_to_amino': ix_to_amino,
        'batch_size': batch_size,
        'max_len': max_len,
        'input_dim': 20,  # 21
        'encoding_dim': encoding_dim,
        'model_state_dict': model.state_dict(),
    }, f'{save_dir}/lstm_model_encoding_dim={encoding_dim}.pt')


def main_nni(stable_params):
    datafile = stable_params['datafile']
    SEQ = stable_params["SEQ"]
    HLA_or_Pep = stable_params["HLA_or_Pep"]
    epochs = stable_params["EPOCHS"]
    encoding_dim = stable_params["ENCODING_DIM"]  # number of dimensions in the embedded space

    nni_params = nni.get_next_parameter()
    batch_size = nni_params["batch_size"]
    lr = nni_params["lr"]
    weight_decay = nni_params["weight_decay"]
    num_layers_enc = nni_params["num_layers_enc"]
    num_layers_dec = nni_params["num_layers_dec"]
    dropout_enc = nni_params["dropout_enc"]
    dropout_dec = nni_params["dropout_dec"]

    optim_params = {'lr': lr, 'weight_decay': weight_decay}
    model_params = {'num_layers_enc': num_layers_enc, 'num_layers_dec': num_layers_dec,
                    'dropout_enc': dropout_enc, 'dropout_dec': dropout_dec}

    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    amino_to_ix = {amino: index for index, amino in enumerate(amino_acids)}
    # ix_to_amino = {index: amino for index, amino in enumerate(amino_acids)}

    print('loading data')
    sequences = load_all_data(datafile, SEQ)
    print('finished loading')

    vecs_data = data_preprocessing_lstm(sequences, amino_to_ix)

    # train + validation sets
    train, validation, _, _ = train_test_split(vecs_data, vecs_data, test_size=0.2)

    if HLA_or_Pep == 'HLA':  # all sequences have same len
        train_clones = convert2tensor(train)
        validation_clones = convert2tensor(validation)

        train_batches = split_to_batches_1clone(train_clones, batch_size)
        validation_batches = split_to_batches_1clone(validation_clones, batch_size)

    elif HLA_or_Pep == 'Pep':  # variable len
        train_clones = lengths_clones(train)
        validation_clones = lengths_clones(validation)

        train_batches = split_to_batches(train_clones, batch_size)
        validation_batches = split_to_batches(validation_clones, batch_size)

    else:
        print("Error: Sequences type has to be 'HLA' or 'Pep'")
        return

    max_len = find_max_len([seq for seq in sequences])
    print('max_len:', max_len)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _, _, _, _ = train_model(train_batches, validation_batches, optim_params, model_params,
                             encoding_dim=encoding_dim, epochs=epochs, device=device,
                             early_stopping=True, is_nni=True)


if __name__ == '__main__':
    with open('Parameters/AE_lstm_parameters.json') as f:
        PARAMETERS = json.load(f)

    if not PARAMETERS['IS_NNI']:
        main(PARAMETERS)
    else:
        main_nni(PARAMETERS)

