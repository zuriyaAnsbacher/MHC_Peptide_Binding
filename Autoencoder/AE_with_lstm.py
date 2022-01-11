import os
import csv
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from random import shuffle
from sklearn.model_selection import train_test_split
import torch.optim as optim
from tqdm import tqdm
import random
import pickle
import matplotlib.pyplot as plt
import copy
from pathlib import Path
from numpy import array
from collections import defaultdict


# (1) Encoder
class Encoder(nn.Module):
    def __init__(self, num_features, embedding_size):
        super().__init__()

        # self.seq_len = seq_len
        self.num_features = num_features  # The number of expected features(= dimension size) in the input x
        self.embedding_size = embedding_size  # the number of features in the embedded points of the inputs' number of features
        self.LSTM1 = nn.LSTM(num_features, embedding_size, num_layers=1, batch_first=True)

    def forward(self, x):
        # Inputs: input, (h_0, c_0). -> If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        x = x.view(x.size(0), -1, self.num_features)
        seq_len = x.size(1)
        self.LSTM1.flatten_parameters()
        x, (hidden_state, cell_state) = self.LSTM1(x)
        x = x[:, -1, :]
        return x, seq_len


# (2) Decoder
class Decoder(nn.Module):
    def __init__(self, num_features, output_size):
        super().__init__()

        # self.seq_len = seq_len
        self.num_features = num_features
        self.hidden_size = (2 * num_features)
        self.output_size = output_size
        self.LSTM1 = nn.LSTM(
            input_size=num_features,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, x, seq_len):
        x = x.unsqueeze(1).repeat(1, seq_len, 1)
        self.LSTM1.flatten_parameters()
        x, (hidden_state, cell_state) = self.LSTM1(x)
        x = self.fc(x)
        x = F.softmax(x, dim=2)
        x = x.view(x.size(0), -1)
        return x


# (3) Autoencoder : putting the encoder and decoder together
class LSTM_AE(nn.Module):
    def __init__(self, num_features, encoding_dim):
        super().__init__()

        self.num_features = num_features
        self.encoding_dim = encoding_dim

        self.encoder = Encoder(self.num_features, self.encoding_dim)
        self.decoder = Decoder(self.encoding_dim, self.num_features)

    def forward(self, x):
        # torch.manual_seed(0)
        encoded, seq_len = self.encoder(x)
        decoded = self.decoder(encoded, seq_len)
        return encoded, decoded


def loss_func(x, y):
    MSE = F.mse_loss(x, y, reduction='sum')
    return MSE


def load_all_data(path, seq, samples_count=1000):
    all_data = []
    for directory, subdirectories, files in os.walk(path):
        for file in tqdm(files):
            df = pd.read_csv(os.path.join(directory, file))
            data = list(df[seq])
            random.shuffle(data)
            all_data += data
    all_data = list(set(all_data))  # maybe unnecessary, because I removed duplicates in data processing
    # random.shuffle(all_data)  # if data contains only 1 file, this shuffle is unnecessary
    return all_data


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


def train_model(batches, validation_batches, max_len, encoding_dim, epochs, device, early_stopping=False):
    model = LSTM_AE(num_features=20, encoding_dim=encoding_dim)
    model.to(device)
    loss_function = loss_func
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    train_loss_list = list()
    val_loss_list = list()
    min_loss = float('inf')
    counter = 0
    best_model = 'None'
    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1} / {epochs}')
        train_loss = train_epoch(batches, model, loss_function, optimizer, device)
        train_loss_list.append(train_loss)
        # val_loss = run_validation(model, validation_batches, loss_function, max_len, vgene_dim, device)
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


def evaluate(save_path, batches, model, ix_to_amino, max_len, device, EPOCHS, set_string):
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


def plot_loss(train, val, num_epochs, save_dir, real_epochs):
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


def main():
    # DONT FORGET TO CHANGE THE VARIANCE IN THE LOSS FUNCTION ACCORDING TO THE DATASET
    with open('AE_parameters.json') as f:
        parameters = json.load(f)

    root = parameters["root"]  # path to data set
    version = parameters["version"]
    save_path = root + '_LSTM_Results'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_dir = f'{save_path}/LSTM_version{version}'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # replace the headers with those in the 'csv' file, put 'None' if missing
    DEL_S = parameters["DEL_S"]  # which delimiter to use to separate headers
    SEQ = parameters["SEQ"]
    HLA_or_Pep = parameters["HLA_or_Pep"]

    # CDR3_S = parameters["CDR3_S"]  # header of amino acid sequence of CDR3
    # V_S = parameters["V_S"]  # header of V gene, 'None' for data set with no resolved V

    EPOCHS = parameters["EPOCHS"]  # number of epochs for each model
    # ENCODING_DIM = parameters["ENCODING_DIM"]  # number of dimensions in the embedded space
    # SAMPLES_COUNT = parameters["SAMPLES_COUNT"]

    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    amino_to_ix = {amino: index for index, amino in enumerate(amino_acids)}
    ix_to_amino = {index: amino for index, amino in enumerate(amino_acids)}
    batch_size = 50
    print('loading data')
    # tcrs = load_all_data(root, CDR3_S, V_S, DEL_S, SAMPLES_COUNT)
    sequences = load_all_data(root, SEQ)
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
    # vgene_dict = get_all_vgenes_dict(root, V_S, DEL_S)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, train_loss_list, val_loss_list, num_epochs = train_model(
        train_batches, validation_batches, max_len,
        encoding_dim=encoding_dim, epochs=EPOCHS,
        device=device, early_stopping=True)
    plot_loss(train_loss_list, val_loss_list, num_epochs, save_dir, EPOCHS)
    evaluate(save_dir, train_batches, model, ix_to_amino, max_len, device, EPOCHS, 'Train')
    evaluate(save_dir, test_batches, model, ix_to_amino, max_len, device, EPOCHS, 'Test')

    # add embedding dim to torch.save
    torch.save({
        'amino_to_ix': amino_to_ix,
        'ix_to_amino': ix_to_amino,
        'batch_size': batch_size,
        'max_len': max_len,
        'input_dim': 20,  # 21
        'encoding_dim': encoding_dim,
        'model_state_dict': model.state_dict(),
    }, f'{save_dir}/lstm_model_encoding_dim={encoding_dim}.pt')


if __name__ == '__main__':
    encoding_dim = 20
    main()
