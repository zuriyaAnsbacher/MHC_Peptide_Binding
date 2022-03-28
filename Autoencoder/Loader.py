import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class HLAPepDataset(Dataset):
    def __init__(self, data, pep_header, hla_header, label_header, amino_pos_to_num):
        self.data = data  # DataFrame

        self.pep_header = pep_header
        self.hla_header = hla_header
        self.label_header = label_header

        self.amino_pos_to_num = amino_pos_to_num
        self.amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
        self.amino_to_ix = {amino: index for index, amino in enumerate(self.amino_acids)}
        self.one_hot_map = self.get_one_hot_map()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        hla = self.data[self.hla_header][index]
        pep = self.data[self.pep_header][index]
        label = self.data[self.label_header][index]

        hla = self.convert_to_numeric(hla)
        pep = self.convert_to_one_hot(pep)

        sample = (pep, hla, label)
        return sample

    def get_one_hot_map(self):
        one_hot = {a: [0] * 20 for a in self.amino_to_ix.keys()}
        for key in one_hot:
            one_hot[key][self.amino_to_ix[key]] = 1
        return one_hot

    def convert_to_numeric(self, seq):
        vec = []
        for i in range(len(seq)):
            amino = seq[i]
            pair = (amino, i)
            vec.append(self.amino_pos_to_num[pair])
        return vec

    def convert_to_one_hot(self, seq):
        vec = []
        for s in seq:
            vec += self.one_hot_map[s]
        return vec

    # maybe unnecessary, because we just convert to tensors, which DatLoader know to do
    # usually, we implement 'collate' if we need to add padded, for example.
    def collate(self, batch):
        pep, hla, label = zip(*batch)
        lst = list()
        lst.append(torch.Tensor(pep))
        lst.append(torch.Tensor(hla))
        lst.append(torch.Tensor(label))
        return lst


class HLAPepDataset_2Labels(Dataset):
    def __init__(self, data, headers, hla_oneHot, amino_pos_to_num, concat_type):
        self.data = data  # DataFrame

        pep_header, hla_name_header, hla_seq_header, binary_header, cont_header, flag_header = headers
        self.pep_header = pep_header
        self.hla_name_header = hla_name_header
        self.hla_seq_header = hla_seq_header
        self.binary_header = binary_header
        self.cont_header = cont_header
        self.flag_header = flag_header
        self.concat_type = concat_type

        self.amino_pos_to_num = amino_pos_to_num
        self.amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
        self.amino_to_ix = {amino: index for index, amino in enumerate(self.amino_acids)}
        self.pep_oneHot = self.get_oneHot_map_pep()

        self.hla_oneHot = hla_oneHot
        self.common_hla_oneHot = self.get_oneHot_common_hla() if self.concat_type != 'None' else 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pep = self.data[self.pep_header][index]
        hla_name = self.data[self.hla_name_header][index]
        hla_seq = self.data[self.hla_seq_header][index]
        binary = float(self.data[self.binary_header][index])
        continuous = float(self.data[self.cont_header][index])
        flag = float(self.data[self.flag_header][index])

        hla = self.convert_to_numeric(hla_seq)
        pep = self.convert_to_one_hot(pep, seq_type='pep')
        continuous = np.log(float(continuous) + 1/np.e) if (flag == 1) else continuous # epsilon=1/e (for log(0+eps)=-1)
        hla_oneHot = self.convert_to_one_hot(hla_name, seq_type='hla') if self.concat_type != 'None' else 0

        sample = (pep, hla, binary, continuous, flag, hla_oneHot)
        return sample

    def get_oneHot_map_pep(self):
        one_hot = {a: [0] * 20 for a in self.amino_to_ix.keys()}
        for key in one_hot:
            one_hot[key][self.amino_to_ix[key]] = 1
        return one_hot

    def get_oneHot_common_hla(self):
        one_hot = {hla_name: [0] * (len(self.hla_oneHot)) for hla_name in self.hla_oneHot.keys()}
        for key in one_hot:
            one_hot[key][self.hla_oneHot[key]] = 1
        one_hot['else'] = [0] * (len(self.hla_oneHot))  # 0 vector for not common alleles
        return one_hot

    def convert_to_numeric(self, seq):
        vec = [0] * len(seq)
        for i in range(len(seq)):
            amino = seq[i]
            pair = (amino, i)
            vec[i] = self.amino_pos_to_num[pair]
            # vec.append(self.amino_pos_to_num[pair])
        return vec

    def convert_to_one_hot(self, seq, seq_type):
        vec = []

        if seq_type == 'pep':
            for s in seq:
                vec += self.pep_oneHot[s]

        elif seq_type == 'hla':
            if seq in self.common_hla_oneHot:
                vec = self.common_hla_oneHot[seq]
            else:
                vec = self.common_hla_oneHot['else']  # vector 0
        return vec

    @staticmethod
    def oneHot_to_scalar(oneHot_lst):
        def to_scalar(x):
            if x.count(1) == 0:  # not common allele, so one_hot is 0-vector. so get fixed val for embedding
                return len(x)
            else:  # common allele. val for embedding is its index in the one-hot
                return x.index(1)
        return [to_scalar(item) for item in oneHot_lst]

    def collate(self, batch):
        pep, hla, binary, continuous, flag, hla_oneHot = zip(*batch)
        lst = list()
        lst.append(torch.Tensor(pep))
        lst.append(torch.Tensor(hla))
        lst.append(torch.Tensor(binary))
        lst.append(torch.Tensor(continuous))
        lst.append(torch.Tensor(flag))
        lst.append(torch.Tensor(hla_oneHot))
        if self.concat_type != 'None':
            lst.append(torch.Tensor(self.oneHot_to_scalar(hla_oneHot)))
        else:
            lst.append(torch.Tensor(hla_oneHot))
        return lst


def check():

    def create_dict_aa_position(max_len):
        pos_to_num = dict()
        count = 1
        for amino in amino_acids:
            for pos in range(max_len):
                pair = (amino, pos)
                pos_to_num[pair] = count
                count += 1
        return pos_to_num

    data = pd.read_csv('HLA_Pep/mhc_pep05_with_neg_PepLen9.csv')
    hla_header = 'HLA Seq'
    pep_header = 'Pep'
    label_header = 'Binding'

    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    amino_pos_to_num = create_dict_aa_position(max_len=183)

    dataset = HLAPepDataset(data, pep_header, hla_header, label_header, amino_pos_to_num)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=dataset.collate)

    for batch in data_loader:
        print(batch)


# check()
