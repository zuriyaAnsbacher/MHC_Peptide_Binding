import csv
import json
import pandas as pd
import numpy as np
import random


# called once for creating seq_dict
def create_seq_dict(json_file):  # from files A.csv, B.csv, C.csv.
    seq_dict = {}
    # create dict that map between seq name and seq seq
    for HLA in ['A', 'B', 'C']:
        with open(f'../Data/HLA-I Seq/{HLA}.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader, None)  # skip headers
            for line in reader:
                hla_name = line[0]
                hla_seq = line[1]
                seq_dict[hla_name] = hla_seq

    with open(json_file, 'w') as file:
        json.dump(seq_dict, file)


# called once for yoram. check HLA with same a.a. sequences
def different_hlaName_with_same_hlaSeq(seq_json, output_file):
    with open(seq_json, 'r') as file:
        seq_dict = json.load(file)

    opposite_dict = {}
    for key1, value1 in seq_dict.items():
        for key2, value2 in seq_dict.items():
            if value1 == value2 and key1 != key2:
                if value1 not in opposite_dict:
                    opposite_dict[value1] = [key1, key2]
                else:
                    if key1 not in opposite_dict[value1]:
                        opposite_dict[value1].append(key1)
                    if key2 not in opposite_dict[value1]:
                        opposite_dict[value1].append(key2)

    with open(output_file, 'w') as file:
        writer = csv.writer(file)
        for key, value in opposite_dict.items():
            writer.writerow([key, value])


def clear_and_extract_data(df):
    df = df[df[10] == 'Linear peptide']
    df = df[df[19] == 'Homo sapiens']
    df = df[df[95].str.contains(':')]  # for removing values such 'HLA class I', 'HLA-

    pep = df[11]
    hla = df[95]
    binding = df[83]

    new_df = pd.concat([pep, hla, binding], axis=1)
    new_df.columns = ['Pep', 'HLA', 'Binding']

    return new_df


def pep_valid(df):
    # 'OVYLVMEL + OX(6M)' -> 'OVYLVMEL' (for remove peptides like this, use 'remove_invalid_pep()')
    df['Pep'] = pd.DataFrame(df['Pep']).applymap(lambda x: x.split()[0])
    # df = df[df['Pep'].apply(lambda x: x.split()[0])]

    # remove peptide with len < threshold_min or len > threshold_max
    threshold_min = 7
    threshold_max = 11
    df = df[df['Pep'].apply(lambda x: threshold_min <= len(x) <= threshold_max)]

    # remove peptides which contains characters that are not amino acids letters
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']

    def valid_pep(col):
        if any([aa not in amino_acids for aa in col]):
            return False
        return True

    df = df[df['Pep'].apply(valid_pep)]

    return df


def drop_nan(df):
    df.dropna(inplace=True)
    return df


def split_to_classes(df):  # classI, classII, and data with '/' (2 options for HLA, only in classII)
    df_slash = df[df["HLA"].str.contains("/")]
    df_classI = df[df['HLA'].str.contains('A\*|B\*|C\*')]
    df_classII = df[df['HLA'].str.contains('DR|DQ|DP') & ~df["HLA"].str.contains("/")]

    return df_slash, df_classI, df_classII


def add_sequences(df, seq_json):
    def name_to_seq(f):
        return seq_dict[f['HLA'].split('-')[1]]

    with open(seq_json, 'r') as file:
        seq_dict = json.load(file)

    df['HLA Seq'] = df.apply(name_to_seq, axis=1)
    df = df[['Pep', 'HLA Seq', 'Binding']]

    return df


def convert_binding_to_binary(df):
    def convert(DF):
        if DF['Binding'] in ['Positive', 'Positive-High', 'Positive-Intermediate']:
            return 1
        elif DF['Binding'] in ['Negative', 'Positive-Low']:
            return 0
        else:
            print('Error')

    df['Binding'] = df.apply(convert, axis=1)
    return df


def split_existing_examples(df):
    df_pairs = df.drop(['Binding'], axis=1)
    df_pairs = df_pairs.apply(tuple, axis=1)

    pos_size = len(df[df['Binding'] == 1])
    neg_size = len(df[df['Binding'] == 0])
    return list(df_pairs), pos_size, neg_size


def create_new_neg(all_pairs, pos_size):
    examples = []
    i = 0
    peps = [p for (p, h) in all_pairs]
    hlas = [h for (p, h) in all_pairs]
    while i < 0.2 * pos_size:  # ~ 85% pos, 15% neg. (existing neg samples in data is insignificant num)
        single_pep = random.choice(peps)
        single_hla = random.choice(hlas)
        if (single_pep, single_hla) not in all_pairs and \
                (single_pep, single_hla, 0) not in examples:
            examples.append([single_pep, single_hla, 0])
            i += 1
    return examples


def create_negative_samples(df):
    all_pairs, pos_size, neg_size = split_existing_examples(df)
    new_neg = create_new_neg(all_pairs, pos_size)
    new_neg = pd.DataFrame(np.array(new_neg), columns=['Pep', 'HLA Seq', 'Binding'])
    all_data = pd.concat([df, new_neg], axis=0)

    return all_data


def process():
    part = '05'
    input_file = f'../Data/IEDB_orig/MHC_and_Pep/mhc_ligand_full_{part}.csv'
    output_file = f'../Data/IEDB_processed/MHC_and_Pep/mhc_pep{part}_15perc_neg_PepLen7_11.csv'
    seq_json = '../Data/HLA-I Seq/dict_seq.json'

    df = pd.read_csv(input_file, header=None)
    df = clear_and_extract_data(df)
    df = pep_valid(df)
    df = drop_nan(df)
    df_slash, df_classI, df_classII = split_to_classes(df)
    df_classI = add_sequences(df_classI, seq_json)
    df_classI = convert_binding_to_binary(df_classI)
    df_classI = create_negative_samples(df_classI)

    df_classI.to_csv(output_file, index=False)


if __name__ == '__main__':
    process()
