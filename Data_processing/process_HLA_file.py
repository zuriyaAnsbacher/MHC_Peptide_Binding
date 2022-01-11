import pandas as pd


def process():
    # output_file = '../Data/HLA-I Seq/hla_all_seq.csv'
    output_file = '../Data/HLA-I Seq/all_seq_A.csv'

    new_df = None
    # for HLA in ['A', 'B', 'C']:
    for HLA in ['A']:
        df = pd.read_csv(f'../Data/HLA-I Seq/{HLA}.csv')
        hla_seq = df['HLA_SEQ']
        if HLA == 'A':
            new_df = hla_seq
        else:
            new_df = pd.concat([new_df, hla_seq], axis=0)

    new_df.drop_duplicates(inplace=True)
    new_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    process()
