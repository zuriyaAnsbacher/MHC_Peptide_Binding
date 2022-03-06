import os
import json
import csv
import nni
import copy
import pandas as pd
import random
from random import shuffle
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from Models import CNN_AE


def load_all_data(datafile, SEQ):
    df = pd.read_csv(datafile)
    data = list(df[SEQ])
    random.shuffle(data)
    data = list(set(data))
    return data


def find_max_len(SEQs):
    return max([len(seq) for seq in SEQs])


def create_dict_aa_position(amino_acids, max_len):
    amino_pos_to_num = dict()
    count = 1
    for amino in amino_acids:
        for pos in range(max_len):
            pair = (amino, pos)
            amino_pos_to_num[pair] = count
            count += 1
    # amino = 'X'
    # for pos in range(max_len):
    #     pair = (amino, pos)
    #     amino_pos_to_num[pair] = count

    return amino_pos_to_num


def pad_one_hot(seq, amino_to_ix, amino_pos_to_num, max_len):
    padding = torch.zeros(1, max_len)
    padding_for_loss = torch.zeros(1, max_len * 20)  # max_len * (20 + 1)

    for i in range(len(seq)):
        amino = seq[i]
        pair = (amino, i)
        padding[0][i] = amino_pos_to_num[pair]
        padding_for_loss[0][i * 20 + amino_to_ix[amino]] = 1  # (20 + 1)

    return padding, padding_for_loss


def get_batches(sequences, amino_to_ix, amino_pos_to_num, batch_size, max_len):
    batches = []
    batches_for_loss = []
    index = 0

    # go over all the data
    while index < len(sequences) // batch_size * batch_size:
        batch_seq = sequences[index:index + batch_size]
        index += batch_size

        padded_seqs = torch.zeros((batch_size, max_len))
        padded_seqs_for_loss = torch.zeros((batch_size, max_len * 20))  # (20 + 1)

        for i in range(batch_size):
            seq = batch_seq[i]
            padded_one, padded_one_for_loss = pad_one_hot(seq, amino_to_ix,
                                                          amino_pos_to_num, max_len)
            padded_seqs[i] = padded_one
            padded_seqs_for_loss[i] = padded_one_for_loss

        batches.append(padded_seqs)
        batches_for_loss.append(padded_seqs_for_loss)

    return batches, batches_for_loss


def loss_func(x, y):
    MSE = F.mse_loss(x, y, reduction='sum')
    return MSE


def train_epoch(batches, batches_for_loss,
                model, loss_function, optimizer, device):
    model.train()
    zip_batches = list(zip(batches, batches_for_loss))
    shuffle(zip_batches)
    batches, batches_for_loss = zip(*zip_batches)
    total_loss = 0
    for i in tqdm(range(len(batches))):
        padded_seqs = batches[i]
        padded_seqs_for_loss = batches_for_loss[i]
        padded_seqs = padded_seqs.to(device)
        padded_seqs_for_loss = padded_seqs_for_loss.to(device)

        model.zero_grad()
        mu, pred = model(padded_seqs)

        loss = loss_function(pred, padded_seqs_for_loss)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"train loss: {total_loss / len(batches)}")
    return total_loss / len(batches)


def run_validation(model, batches, batches_for_loss, loss_function, device):
    model.eval()
    zip_batches = list(zip(batches, batches_for_loss))
    shuffle(zip_batches)
    batches, batches_for_loss = zip(*zip_batches)
    total_loss_val = 0
    for i in range(len(batches)):
        with torch.no_grad():
            padded_seqs = batches[i]
            padded_seqs_for_loss = batches_for_loss[i]
            padded_seqs = padded_seqs.to(device)
            padded_seqs_for_loss = padded_seqs_for_loss.to(device)

            mu, pred = model(padded_seqs)

            loss = loss_function(pred, padded_seqs_for_loss)

            total_loss_val += loss.item()

    print(f"val loss: {total_loss_val / len(batches)}")
    return total_loss_val / len(batches)


def run_model(train_batches, validation_batches,
              train_batches_for_loss, validation_batches_for_loss,
              max_len, optim_params, model_params, encoding_dim,
              epochs, batch_size, device, early_stopping=False, is_nni=False):

    model = CNN_AE(max_len=max_len, encoding_dim=encoding_dim, batch_size=batch_size, model_params=model_params)
    model.to(device)
    loss_function = loss_func
    optimizer = optim.Adam(model.parameters(), lr=optim_params['lr'],  weight_decay=optim_params['weight_decay'])
    train_loss_list = []
    val_loss_list = []
    min_loss = float('inf')
    counter = 0
    best_model = 'None'

    for epoch in range(1, epochs + 1):
        print(f'Epoch: {epoch} / {epochs}')
        train_loss = train_epoch(train_batches, train_batches_for_loss, model,
                                 loss_function, optimizer, device)
        val_loss = run_validation(model, validation_batches, validation_batches_for_loss,
                                  loss_function, device)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        if val_loss < min_loss:
            min_loss = val_loss
            counter = 0
            best_model = copy.deepcopy(model)
        elif early_stopping and counter == 30:
            break
        else:
            counter += 1

        if is_nni:
            nni.report_intermediate_result(val_loss)

    if is_nni:
        nni.report_final_result(min_loss)

    num_epochs = epoch
    if best_model == 'None':
        best_model = copy.deepcopy(model)

    return best_model, train_loss_list, val_loss_list, num_epochs


def plot_loss(train, val, num_epochs, save_dir):
    epochs = [e for e in range(1, num_epochs + 1)]
    label1 = 'Train'
    label2 = 'Validation'
    plt.figure(2)
    plt.plot(epochs, train, 'bo', color='mediumaquamarine', label=label1)
    plt.plot(epochs, val, 'b', color='cornflowerblue', label=label2)
    plt.xlabel('Epoch')
    plt.ylabel('Average loss')
    plt.legend()
    plt.savefig(f'{save_dir}/mse_loss.pdf')
    plt.close()


def read_pred(preds, ix_to_amino):
    batch_seqs = []
    for seq in preds:
        c_index = torch.argmax(seq, dim=1)
        s = ''
        for index in c_index:
            # if ix_to_amino[index.item()] == 'X':
            #     break
            s += ix_to_amino[index.item()]
        batch_seqs.append(s)
    return batch_seqs


def count_mistakes(true_seq, pred_seq):
    mis = 0
    for i in range(min(len(true_seq), len(pred_seq))):
        if not true_seq[i] == pred_seq[i]:
            mis += 1
    return mis


def evaluate(model, batches, batches_for_loss, ix_to_amino, max_len,
             device, encoding_dim, embedding_dim, save_dir):
    model.eval()
    print('Evaluating')
    zip_batches = list(zip(batches, batches_for_loss))
    shuffle(zip_batches)
    batches, batches_for_loss = zip(*zip_batches)
    acc = 0
    acc_1mis = 0
    acc_2mis = 0
    acc_3mis = 0
    count = 0
    for i in tqdm(range(len(batches))):
        with torch.no_grad():
            padded_seqs = batches[i]
            padded_seqs_for_loss = batches_for_loss[i]

            # extract amino acids sequences form the padded sequences
            true_seqs = read_pred(padded_seqs_for_loss.view(-1, max_len, 20), ix_to_amino)  # 20 + 1

            padded_seqs = padded_seqs.to(device)
            mu, pred = model(padded_seqs)

            pred_seqs = read_pred(pred.view(-1, max_len, 20), ix_to_amino)  # 20 + 1

            for j in range(len(batches[i])):
                count += 1
                if len(true_seqs[j]) != len(pred_seqs[j]):
                    continue
                mis = count_mistakes(true_seqs[j], pred_seqs[j])
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

    print('acc:', acc)
    print('acc_1mis:', acc_1mis)
    print('acc_2mis:', acc_2mis)
    print('acc_3mis:', acc_3mis)

    with open(f'{save_dir}/test_accuracy_embedding={embedding_dim}_encoding={encoding_dim}.csv', 'w') as csvfile:
        fieldnames = ['Accuracy', '1 Mismatch', '2 Mismatches', '3 Mismatches']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Accuracy': str(acc), '1 Mismatch': str(acc_1mis),
                         '2 Mismatches': str(acc_2mis), '3 Mismatches': str(acc_3mis)})


def main(parameters):
    root = parameters["root"]
    datafile = parameters["datafile"]
    version = parameters["version"]
    save_path = root + '_CNN_Results'
    save_dir = f'{save_path}/CNN_version{version}'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    SEQ = parameters["SEQ"]
    epochs = parameters["EPOCHS"]
    encoding_dim = parameters["ENCODING_DIM"]
    batch_size = parameters["BATCH_SIZE"]
    embedding_dim = parameters["EMBEDDING_DIM"]

    optim_params = {'lr': 1e-4, 'weight_decay': 0}
    model_params = {'embedding_dim': embedding_dim,
                    'filter1': 16, 'filter2': 32,
                    'kernel1': 2, 'kernel2': 2,
                    'dropout_enc': 0.2, 'dropout_dec': 0.1}

    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    # amino_to_ix = {amino: index for index, amino in enumerate(amino_acids + ['X'])}
    # ix_to_amino = {index: amino for index, amino in enumerate(amino_acids + ['X'])}
    amino_to_ix = {amino: index for index, amino in enumerate(amino_acids)}
    ix_to_amino = {index: amino for index, amino in enumerate(amino_acids)}

    print('loading data')
    SEQs = load_all_data(datafile, SEQ)
    print('finished loading')

    train, test, _, _ = train_test_split(SEQs, SEQs, test_size=0.2)
    train, validation, _, _ = train_test_split(train, train, test_size=0.15)

    max_len = find_max_len(SEQs)
    print('max len:', max_len)

    amino_pos_to_num = create_dict_aa_position(amino_acids, max_len)

    train_batches, train_batches_for_loss = get_batches(train, amino_to_ix, amino_pos_to_num,
                                                        batch_size, max_len)
    test_batches, test_batches_for_loss = get_batches(test, amino_to_ix, amino_pos_to_num,
                                                      batch_size, max_len)
    validation_batches, validation_batches_for_loss = get_batches(validation, amino_to_ix,
                                                                  amino_pos_to_num, batch_size, max_len)
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    model, train_loss_list, val_loss_list, num_epochs = \
        run_model(train_batches, validation_batches, train_batches_for_loss, validation_batches_for_loss,
                  max_len, optim_params, model_params, encoding_dim=encoding_dim, epochs=epochs,
                  batch_size=batch_size, device=device, early_stopping=True)

    plot_loss(train_loss_list, val_loss_list, num_epochs, save_dir)
    evaluate(model, test_batches, test_batches_for_loss, ix_to_amino, max_len,
             device, encoding_dim, embedding_dim, save_dir)

    torch.save({
        'amino_to_ix': amino_to_ix,
        'ix_to_amino': ix_to_amino,
        'batch_size': batch_size,
        'amino_pos_to_num': amino_pos_to_num,
        'max_len': max_len,
        'input_dim': 20,  # 20 + 1
        'encoding_dim': encoding_dim,
        'embedding_dim': embedding_dim,
        'model_state_dict': model.state_dict(),
    }, f'{save_dir}/cnn_embedding={embedding_dim}_encoding={encoding_dim}.pt')


def main_nni(stable_params):

    datafile = stable_params["datafile"]
    SEQ = stable_params["SEQ"]
    epochs = stable_params["EPOCHS"]
    encoding_dim = stable_params["ENCODING_DIM"]

    nni_params = nni.get_next_parameter()
    batch_size = nni_params["batch_size"]
    lr = nni_params["lr"]
    weight_decay = nni_params["weight_decay"]
    embedding_dim = nni_params["embedding_dim"]
    filter1 = nni_params["filter1"]
    filter2 = nni_params["filter2"]
    kernel1 = nni_params["kernel1"]
    kernel2 = nni_params["kernel2"]
    dropout_enc = nni_params["dropout_enc"]
    dropout_dec = nni_params["dropout_dec"]

    optim_params = {'lr': lr, 'weight_decay': weight_decay}
    model_params = {'embedding_dim': embedding_dim,
                    'filter1': filter1, 'filter2': filter2,
                    'kernel1': kernel1, 'kernel2': kernel2,
                    'dropout_enc': dropout_enc, 'dropout_dec': dropout_dec}

    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    amino_to_ix = {amino: index for index, amino in enumerate(amino_acids)}

    print('loading data')
    sequences = load_all_data(datafile, SEQ)
    print('finished loading')

    train, validation, _, _ = train_test_split(sequences, sequences, test_size=0.2)

    max_len = find_max_len(sequences)
    print('max len:', max_len)

    amino_pos_to_num = create_dict_aa_position(amino_acids, max_len)

    train_batches, train_batches_for_loss = get_batches(train, amino_to_ix, amino_pos_to_num,
                                                        batch_size, max_len)
    validation_batches, validation_batches_for_loss = get_batches(validation, amino_to_ix,
                                                                  amino_pos_to_num, batch_size, max_len)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _, _, _, _ = run_model(train_batches, validation_batches, train_batches_for_loss, validation_batches_for_loss,
                           max_len, optim_params, model_params, encoding_dim=encoding_dim, epochs=epochs,
                           batch_size=batch_size, device=device, early_stopping=True, is_nni=True)


if __name__ == '__main__':
    with open('Parameters/AE_CNN_parameters.json') as f:
        PARAMETERS = json.load(f)

    if not PARAMETERS['IS_NNI']:
        main(PARAMETERS)
    else:
        main_nni(PARAMETERS)
