import numpy as np
from random import shuffle
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils.data import Dataset

# code based on:
#   "https://gist.github.com/TrentBrick/bac21af244e7c772dc8651ab9c58328c",
#   "https://github.com/shenkev/Pytorch-Sequence-Bucket-Iterator"


class SamplerByLength(Sampler):
    """organize data to batches such that each batch contains samples with same length of peptides"""

    def __init__(self, data_source, bucket_boundaries, batch_size):
        ind_n_len = []
        for i in range(len(data_source)):
            p = data_source[i]
            ind_n_len.append((i, len(p[0])))  # [(idx_tensor_data, len), ...]

        self.data_source = data_source
        self.ind_n_len = ind_n_len
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size

        self.iter_order = None

    def __iter__(self):
        data_buckets = dict()
        for sample_idx, sample_len in self.ind_n_len:
            if sample_len in data_buckets.keys():
                data_buckets[sample_len].append(sample_idx)
            else:
                data_buckets[sample_len] = [sample_idx]

        for k in data_buckets.keys():
            data_buckets[k] = np.asarray(data_buckets[k])

        iter_list = []
        for samples_same_len in data_buckets.values():
            np.random.shuffle(samples_same_len)
            redundant_samples_num = samples_same_len.shape[0] % self.batch_size
            for i in range(redundant_samples_num):  # delete samples such that samples_same_len % batch_size = 0
                samples_same_len = np.delete(samples_same_len, 0)
            if samples_same_len.shape[0] >= self.batch_size:
                iter_list += (np.array_split(samples_same_len, (samples_same_len.shape[0] // self.batch_size)))
        shuffle(iter_list)

        self.iter_order = iter_list  # for function 'evaluate_splitScoreByFreqHLA'

        for i in iter_list:
            yield i.tolist()

    def __len__(self):
        return len(self.data_source)


# for test 'SamplerByLength' class
class MyDataset(Dataset):
    def __init__(self):
        self.data = [
            ([1, 1], [0], [0]),
            ([1, 1, 1], [1], [1]),
            ([1, 1], [2], [2]),
            ([1, 1, 1, 1], [3], [3]),
            ([1, 1], [4], [4]),
            ([1, 1, 1], [5], [5]),
            ([1, 1], [6], [6]),
            ([1, 1, 1, 1], [7], [7]),
            ([1, 1], [8], [8]),
            ([1, 1, 1], [9], [9])
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return [torch.tensor(self.data[index][0]),
                torch.tensor(self.data[index][1]),
                torch.tensor(self.data[index][2])]


def check():
    np.random.seed(0)
    torch.manual_seed(0)

    bucket_boundaries = [2, 3, 4]

    my_data = MyDataset()

    sampler = SamplerByLength(my_data, bucket_boundaries, 2)

    dataloader = DataLoader(my_data, batch_size=1,
                            batch_sampler=sampler)

    for batch in dataloader:
        print(batch)


# check()