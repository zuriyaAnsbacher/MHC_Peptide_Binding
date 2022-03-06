import csv
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score, r2_score


def evaluate_splitScoreByFreqHLA(model, test_loader, test_df, test_sampler, batch_size, device):

    def checkFreq(ind_sample, ind_batch):
        idx = test_sampler.iter_order[ind_batch][ind_sample]
        hla_name = test_df['HLA name'][idx]
        freq = name2freq[hla_name]
        if freq >= 2944:
            return 'high'
        elif freq <= 87:
            return 'low'
        else:
            return 'med'

    name2freq = {}
    with open('../Data/Common HLA-I/common_A_IEDB_files.csv') as file2:
        reader = csv.reader(file2)
        next(reader)  # skip headers
        for line in reader:
            name2freq[line[0]] = int(line[1])

    model.eval()

    preds1_high, preds2_high = [], []
    preds1_med, preds2_med = [], []
    preds1_low, preds2_low = [], []

    trues1_high, trues2_high = [], []
    trues1_med, trues2_med = [], []
    trues1_low, trues2_low = [], []

    flags_high = []
    flags_med = []
    flags_low = []

    with torch.no_grad():
        for idx_batch, batch in enumerate(test_loader):
            pep, hla, y1, y2, flag, hla_oneHot, emb = batch
            pep, hla, y1, y2, flag, hla_oneHot, emb = pep.to(device), hla.to(device), y1.to(device), y2.to(device), \
                                                      flag.to(device), hla_oneHot.to(device), emb.to(device)

            pred1, pred2 = model(pep, hla, hla_oneHot, emb)
            pred1, pred2 = pred1.view(-1), pred2.view(-1)

            pred1_lst = torch.tensor_split(pred1, batch_size, dim=0)
            y1_lst = torch.tensor_split(y1, batch_size, dim=0)
            pred2_lst = torch.tensor_split(pred2, batch_size, dim=0)
            y2_lst = torch.tensor_split(y2, batch_size, dim=0)
            flag_lst = torch.tensor_split(flag, batch_size, dim=0)

            for idx_sample, (single_pred1, single_y1, single_pred2, single_y2, single_flag) in \
                    enumerate(zip(pred1_lst, y1_lst, pred2_lst, y2_lst, flag_lst)):

                single_pred1 = single_pred1.item()
                single_y1 = single_y1.item()
                single_pred2 = single_pred2.item()
                single_y2 = single_y2.item()
                single_flag = single_flag.item()

                sample_freq = checkFreq(idx_sample, idx_batch)
                if sample_freq == 'high':
                    preds1_high.append(expit(single_pred1))
                    trues1_high.append(single_y1)
                    preds2_high.append(single_pred2)
                    trues2_high.append(single_y2)
                    flags_high.append(single_flag)

                elif sample_freq == 'med':
                    preds1_med.append(expit(single_pred1))
                    trues1_med.append(single_y1)
                    preds2_med.append(single_pred2)
                    trues2_med.append(single_y2)
                    flags_med.append(single_flag)

                elif sample_freq == 'low':
                    preds1_low.append(expit(single_pred1))
                    trues1_low.append(single_y1)
                    preds2_low.append(single_pred2)
                    trues2_low.append(single_y2)
                    flags_low.append(single_flag)

    for case, (trues1, preds1), (trues2, preds2), flags in zip(
            ['high', 'medium', 'low'],
            [(trues1_high, preds1_high), (trues1_med, preds1_med), (trues1_low, preds1_low)],
            [(trues2_high, preds2_high), (trues2_med, preds2_med), (trues2_low, preds2_low)],
            [flags_high, flags_med, flags_low]
    ):
        preds1 = [preds1[i] for i in range(len(flags)) if flags[i] == 0]
        trues1 = [trues1[i] for i in range(len(flags)) if flags[i] == 0]
        preds2 = [preds2[i] for i in range(len(flags)) if flags[i] == 1]
        trues2 = [trues2[i] for i in range(len(flags)) if flags[i] == 1]

        auc = roc_auc_score(trues1, preds1)
        r2 = r2_score(trues2, preds2)
        print(f"Case {case}:\tauc test: ", round(auc, 4), " | ", "r2 test: ", round(r2, 4))

