import pandas as pd

input_file = '../Data/HLA-I Seq/originalABC/A.csv'

df = pd.read_csv(input_file, header=None, delimiter='\t')
df_aa = df[2].str.split(',', expand=True)

for col in df_aa.columns:
    sum_exceptional = 0
    value_count = df_aa[col].value_counts()
    if len(value_count) <= 4:
        # count how many exceptional
        for idx_row, row in enumerate(value_count.index):
            if idx_row > 0:
                sum_exceptional += value_count[row]
        print(f'Column: {col}, Values count: {len(value_count)}, Exceptional: {sum_exceptional}')


