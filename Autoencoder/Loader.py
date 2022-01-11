import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class HLAPepDataset(Dataset):
    def __init__(self, data, pep_header, hla_header, label_header, amino_pos_to_num):
        self.data = data  # DataFrame

        self.hla_header = hla_header
        self.pep_header = pep_header
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
        label = self.data[self.label_header][index]  # check if 0/1 or 'Positive/'..

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






