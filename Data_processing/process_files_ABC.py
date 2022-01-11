import csv


for HLA in ['A', 'B', 'C']:
    input_file = f'../Data/HLA-I Seq/originalABC/{HLA}.csv'
    output_file = f'../Data/HLA-I Seq/{HLA}.csv'

    with open(input_file, 'r') as in_f:
        with open(output_file, 'w') as out_f:
            reader = csv.reader(in_f, delimiter='\t')
            writer = csv.writer(out_f)
            writer.writerow(['HLA_NAME', 'HLA_SEQ'])
            for line in reader:
                hla_name = ''.join([line[0], '*', line[1]])
                hla_seq = ''
                for aa in line[2].split(','):
                    hla_seq += aa[0]
                writer.writerow([hla_name, hla_seq])

