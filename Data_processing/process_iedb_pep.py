import pandas as pd


def clear_and_extract_data(df):
    df = df[df[('Epitope', 'Object Type')] == 'Linear peptide']
    # df = df[df[('Epitope', 'Organism Name')] == 'Homo sapiens']  # if we want only human peptides

    new_df = pd.DataFrame(df[('Epitope', 'Description')])
    new_df.columns = ['Pep']

    return new_df


def pep_valid(df):
    # 'OVYLVMEL + OX(6M)' -> 'OVYLVMEL'
    df = df.applymap(lambda x: x.split()[0])

    # remove peptide with len < threshold_min
    threshold_min = 7
    threshold_max = 11
    df = df[df['Pep'].apply(lambda x: threshold_min <= len(x) <= threshold_max)]

    # remove peptides which contains characters that are not amino acids letters
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']

    def contains_aa_only(col):
        if any([aa not in amino_acids for aa in col]):
            return False
        return True

    df = df[df['Pep'].apply(contains_aa_only)]

    return df


def drop_nan(df):
    df.dropna(inplace=True)
    return df


def process():
    input_file = '../Data/IEDB_orig/Pep/epitope_full_v3.csv'
    output_file = '../Data/IEDB_processed/Pep/epitope_full_v3.csv'

    df = pd.read_csv(input_file, header=[0, 1])
    df = clear_and_extract_data(df)
    df = pep_valid(df)
    df = drop_nan(df)

    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    process()
