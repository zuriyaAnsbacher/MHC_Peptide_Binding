import csv
import json
import pandas as pd
import numpy as np
import random

PEP_HEAD = 'Pep'
HLA_NAME_HEAD = 'HLA name'
HLA_SEQ_HEAD = 'HLA Seq'
BIND_HEAD = 'Binding'
BINARY_BIND_HEAD = 'Binary binding'
CONT_BIND_HEAD = 'Continuous binding'
FLAG_HEAD = 'Flag'


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


def clear_and_extract_data(df, labels_num):
    df = df[df[10] == 'Linear peptide']
    df = df[df[39].str.contains('human')]  # human Host (if want human epitope: it's in column 19)
    df = df[df[95].str.contains(':')]  # for removing values such 'HLA class I', 'HLA-A'
    df = df[~df[95].str.contains('mutant')]  # for removing values such 'A*02:01 K66A mutant'

    pep = df[11]
    hla = df[95]

    if labels_num == 1:  # binary only
        binding = df[83]

        new_df = pd.concat([pep, hla, binding], axis=1)
        new_df.columns = [PEP_HEAD, HLA_NAME_HEAD, BIND_HEAD]

    elif labels_num == 2:  # binary + continuous
        binary_binding = df[83]
        cont_binding = df[85]

        new_df = pd.concat([pep, hla, binary_binding, cont_binding], axis=1)
        new_df.columns = [PEP_HEAD, HLA_NAME_HEAD, BINARY_BIND_HEAD, CONT_BIND_HEAD]

    return new_df


def pep_valid(df):
    # 'OVYLVMEL + OX(6M)' -> 'OVYLVMEL' (for remove peptides like this, use 'remove_invalid_pep()')
    df[PEP_HEAD] = pd.DataFrame(df[PEP_HEAD]).applymap(lambda x: x.split()[0])
    # df = df[df['Pep'].apply(lambda x: x.split()[0])]

    # remove peptide with len < threshold_min or len > threshold_max
    threshold_min = 7
    threshold_max = 11
    df = df[df[PEP_HEAD].apply(lambda x: threshold_min <= len(x) <= threshold_max)]

    # remove peptides which contains characters that are not amino acids letters
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']

    def valid_pep(col):
        if any([aa not in amino_acids for aa in col]):
            return False
        return True

    df = df[df[PEP_HEAD].apply(valid_pep)]

    return df


def drop_nan(df):
    df.dropna(inplace=True)
    return df


def split_to_classes(df, only1allele):  # classI, classII, and data with '/' (2 options for HLA, only in classII)
    df_slash = df[df[HLA_NAME_HEAD].str.contains("/")]
    df_classI = df[df[HLA_NAME_HEAD].str.contains('A\*|B\*|C\*') & ~df[HLA_NAME_HEAD].str.contains("/")]

    if only1allele:
        df_classI = df_classI[df_classI[HLA_NAME_HEAD].str.contains(f'{only1allele}\*')]  # get only A or B or C

    df_classII = df[df[HLA_NAME_HEAD].str.contains('DR|DQ|DP') & ~df[HLA_NAME_HEAD].str.contains("/")]

    return df_slash, df_classI, df_classII


def add_sequences(df, seq_json, labels_num):
    def name_to_seq(f):
        return seq_dict[f[HLA_NAME_HEAD].split('-')[1]]

    with open(seq_json, 'r') as file:
        seq_dict = json.load(file)

    df[HLA_SEQ_HEAD] = df.apply(name_to_seq, axis=1)

    if labels_num == 1:
        df = df[[PEP_HEAD, HLA_NAME_HEAD, HLA_SEQ_HEAD, BIND_HEAD]]
    elif labels_num == 2:
        df = df[[PEP_HEAD, HLA_NAME_HEAD, HLA_SEQ_HEAD, BINARY_BIND_HEAD, CONT_BIND_HEAD, FLAG_HEAD]]

    return df


def convert_binding_to_binary(df, labels_num):
    if labels_num == 1:
        binary_binding_header = BIND_HEAD
    elif labels_num == 2:
        binary_binding_header = BINARY_BIND_HEAD

    def convert(DF):
        if DF[binary_binding_header] in ['Positive', 'Positive-High', 'Positive-Intermediate']:
            return 1
        elif DF[binary_binding_header] in ['Negative', 'Positive-Low']:
            return 0
        else:
            print('Error')

    df[binary_binding_header] = df.apply(convert, axis=1)
    return df


def split_samples_to_labels(df):
    """split samples with 2 labels in datafile to 2 samples (one binary, one continuous),
    and create 'Flag' column, while flag=0 is binary sample and flag=1 is continuous sample"""

    cont_df = df[~df[CONT_BIND_HEAD].isnull()]
    cont_df[BINARY_BIND_HEAD] = cont_df[BINARY_BIND_HEAD].apply(lambda x: -1)
    flag_1 = np.ones(len(cont_df.index))
    flag_1 = pd.DataFrame(flag_1, columns=[FLAG_HEAD])
    cont_df.reset_index(drop=True, inplace=True)
    cont_df = pd.concat([cont_df, flag_1], axis=1)

    df[CONT_BIND_HEAD] = df[CONT_BIND_HEAD].apply(lambda x: -1)
    flag_0 = np.zeros(len(df.index))
    flag_0 = pd.DataFrame(flag_0, columns=[FLAG_HEAD])
    df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, flag_0], axis=1)

    df = pd.concat([df, cont_df], axis=0)
    df.reset_index(drop=True, inplace=True)

    return df


def split_existing_examples(df, labels_num):
    if labels_num == 1:
        df_pairs = df.drop([BIND_HEAD], axis=1)
        binary_binding_header = BIND_HEAD
    elif labels_num == 2:
        df_pairs = df.drop([BINARY_BIND_HEAD, CONT_BIND_HEAD, FLAG_HEAD], axis=1)
        binary_binding_header = BINARY_BIND_HEAD

    df_pairs = df_pairs.apply(tuple, axis=1)

    pos_size = len(df[df[binary_binding_header] == 1])
    neg_size = len(df[df[binary_binding_header] == 0])
    return list(df_pairs), pos_size, neg_size


def create_new_neg(all_pairs, pos_size, labels_num):
    examples = []
    i = 0
    peps = [p for (p, h_name, h_seq) in all_pairs]
    hlas = [(h_name, h_seq) for (p, h_name, h_seq) in all_pairs]
    while len(examples) < 0.3 * len(all_pairs):  # ~ 75% pos, 25% neg. (existing neg samples in data is insignificant num)
        single_pep = random.choice(peps)
        single_hla = random.choice(hlas)

        if labels_num == 1:
            example = [single_pep, single_hla[0], single_hla[1], 0]
        elif labels_num == 2:
            example = [single_pep, single_hla[0], single_hla[1], 0, -1, 0]  # last two: cont binding, flag

        if (single_pep, single_hla[0], single_hla[1]) not in all_pairs and example not in examples:
            examples.append(example)
            i += 1

    return examples


def create_negative_samples(df, labels_num):
    all_pairs, pos_size, neg_size = split_existing_examples(df, labels_num)
    new_neg = create_new_neg(all_pairs, pos_size, labels_num)
    if labels_num == 1:
        new_neg = pd.DataFrame(np.array(new_neg), columns=[PEP_HEAD, HLA_NAME_HEAD, HLA_SEQ_HEAD, BIND_HEAD])
    elif labels_num == 2:
        new_neg = pd.DataFrame(np.array(new_neg), columns=[PEP_HEAD, HLA_NAME_HEAD, HLA_SEQ_HEAD, BINARY_BIND_HEAD, CONT_BIND_HEAD, FLAG_HEAD])
    all_data = pd.concat([df, new_neg], axis=0)

    return all_data


def process():
    part = '00'
    specific_allele = 'A'  # A/B/C or None if want all three of them

    binary_or_binaryContinuous = 'binaryContinuous'
    labels_num = 1 if binary_or_binaryContinuous == 'binary' else 2

    input_file = f'../Data/IEDB_orig/MHC_and_Pep/mhc_ligand_full_{part}.csv'
    output_file = f'../Data/IEDB_processed/MHC_and_Pep/mhc_pep{part}{specific_allele}_25percNeg_Pep7_11_2labels.csv'
    seq_json = '../Data/HLA-I Seq/dict_seq.json'

    if part == '00':
        df = pd.read_csv(input_file, header=None, skiprows=2)
    else:
        df = pd.read_csv(input_file, header=None)

    if binary_or_binaryContinuous == 'binary':
        df = clear_and_extract_data(df, labels_num)
        _, df, _ = split_to_classes(df, only1allele=specific_allele)  # get only HLA class I
        df = pep_valid(df)
        df = drop_nan(df)
        df = convert_binding_to_binary(df, labels_num)
        df = add_sequences(df, seq_json, labels_num)
        df = create_negative_samples(df, labels_num)

        df.to_csv(output_file, index=False)

    elif binary_or_binaryContinuous == 'binaryContinuous':
        df = clear_and_extract_data(df, labels_num)
        _, df, _ = split_to_classes(df, only1allele=specific_allele)
        df = pep_valid(df)
        df = convert_binding_to_binary(df, labels_num)
        df = split_samples_to_labels(df)
        df = drop_nan(df)
        df = add_sequences(df, seq_json, labels_num)
        df = create_negative_samples(df, labels_num)

        df.to_csv(output_file, index=False)

    else:
        print('Error: must be "binary" or "binaryContinuous"')


if __name__ == '__main__':
    process()
