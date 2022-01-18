import pandas as pd
import numpy as np

"""
Distribution:
    File: 00              File: 01            File 02:            File 03:            File 04:            File 05:
    A size:  167320       A size:  46023      A size:  87391      A size:  45457      A size:  96379      A size:  53763
    B size:  142554       B size:  120693     B size:  117957     B size:  46515      B size:  68508      B size:  69778
    C size:  6162         C size:  43230      C size:  46933      C size:  3150       C size:  6419       C size:  1766
"""


def distribution():
    # check approximately distribution of A, B, C in files from IEDB
    # (it's not accurate, because I dont check peptides validity and length range)
    index_file = [str(i) for i in range(6)]
    for index in index_file:
        file = f'../Data/IEDB_orig/MHC_and_Pep/mhc_ligand_full_0{index}.csv'

        if index == '0':
            df = pd.read_csv(file, header=None, skiprows=2)
        else:
            df = pd.read_csv(file, header=None)

        df = df[df[10] == 'Linear peptide']
        df = df[df[39].str.contains('human')]
        df = df[df[95].str.contains(':')]

        df = df[df[95].str.contains('A\*|B\*|C\*') & ~df[95].str.contains("/")]
        dfA = df[df[95].str.contains('A\*')]
        dfB = df[df[95].str.contains('B\*')]
        dfC = df[df[95].str.contains('C\*')]

        print(f'File: 0{index}')
        print('A size: ', len(dfA.index))
        print('B size: ', len(dfB.index))
        print('C size: ', len(dfC.index))


def most_common():
    specific_allele = 'A'
    output_file = f'../Data/Common HLA-I/common_{specific_allele}_IEDB_files.csv'
    all_allele_name = []
    index_file = [str(i) for i in range(6)]

    for index in index_file:
        file = f'../Data/IEDB_orig/MHC_and_Pep/mhc_ligand_full_0{index}.csv'

        if index == '0':
            df = pd.read_csv(file, header=None, skiprows=2)
        else:
            df = pd.read_csv(file, header=None)

        df = df[df[10] == 'Linear peptide']
        df = df[df[39].str.contains('human')]
        df = df[df[95].str.contains(':')]

        dfA = df[df[95].str.contains(f'{specific_allele}\*') & ~df[95].str.contains("/")]
        all_allele_name.extend(list(dfA[95]))

    all_allele_name = pd.DataFrame(np.array(all_allele_name).reshape(-1))

    allele_freq = all_allele_name[0].value_counts()
    print(allele_freq[:30])  # print 30 most common

    allele_freq = pd.DataFrame(allele_freq)
    allele_freq.reset_index(inplace=True)
    allele_freq.to_csv(output_file, index=None, header=['HLA', 'freq'])


distribution()
