import csv

# original files are from: 'https://github.com/ANHIG/IMGTHLA/tree/Latest/alignments'

# processing details:

# take the first sequences (full data), and compare other sequence with it.
# in any position that the current seq has '-' or '*' ,
# or in case that full sequence has '.':
# copy the position from the full sequence to the current seq.
# in cases (there are few.. 2-3 cases) that current seq has '.' and full sequence has letter: copy the letter to the current

# if there are sequences that are shorter than the full sequences (DQA1=255; DQB1=261; DRB1=266),
# remove them from the data


FULL_SEQ = {'DQA1': ['MILNKALLLGALALTTVMSPCGGEDIVADHVASCGVNLYQFYGPSGQYTHEFDGDEEFYVDLERKETAWRWPEFSKFGGFDPQGALRNMAVAK',
                    'HNLNIMIKRYNSTAATNEVPEVTVFSKSPVTLGQPNTLICLVDNIFPPVVNITWLSNGQSVTEGVSETSFLSKSDHSFFKISYLTFLPSADEIYDCKVEH',
                    'WGLDQPLLKHWEPEIPAPMSELTETVVCALGLSVGLVGIVVGTVFIIQGLRSVGASRHQGPL'],
            'DQB1': ['MSWKKSLRIPGDLRVATVTLMLAILSSSLAEGRDSPEDFVYQFKGLCYFTNGTERVRGVTRHIYNREEYVRFDSDVGVYRAVTPQGRPVAEY',
                     'WNSQKEVLEGARASVDRVCRHNYEVAYRGILQRRVEPTVTISPSRTEALNHHNLLICSVTDFYPSQIKVRWFRNDQEETAGVVSTPLIRNGDWTFQILVM',
                     'LEMTPQR...GDVYTCHVEHPSLQSPITVEWRAQSESAQSKMLSGVGGFVLGLIFLGLGLIIRQRSRKG........LLH'],
            'DRB1': ['MVCLKLPGGSCMTALTVTLMVLSSPLALAGDTRPRFLWQLKFECHFFNGTERVR.LLERCIYNQEE.SVRFDSDVGEYRAVTELGRPDAEYWNSQKDLL',
                     'EQRRAAVDTYCRHNYGVGESFTVQRRVEPKVTVYPSKTQPLQHHNLLVCSVSGFYPGSIEVRWFRNGQEEKAGVVSTGLIQNGDWTFQTLVMLETVPRSG',
                     'EVYTCQVEHPSVTSPLTVEWRARSESAQSKMLSGVGGFVLGLLFLGAGLFIYFRNQKGHSGLQPTGFLS']}


def get_curr_seq(curr_seq, full_seq):
    if len(curr_seq) != len(full_seq):
        print(f'Short sequence: len={len(curr_seq)}')
        return None

    curr_seq = [i for i in curr_seq]  # for assign dynamically (in string its not possible)
    for i in range(len(full_seq)):
        if curr_seq[i] == '-' or curr_seq[i] == '*' or full_seq[i] == '.':
            curr_seq[i] = full_seq[i]
        # in specific cases, when there is '.' in the curr seq (but not in the original seq), so treat this like '-'
        elif curr_seq[i] == '.' and full_seq[i] != '.':
            curr_seq[i] = full_seq[i]
    curr_seq = ''.join(curr_seq)  # back to string

    for char in curr_seq:  # validation: all chars are upper letter or dot
        if not(char.isupper() or char == '.'):
            print(f'Check it. seq={curr_seq}')

    return curr_seq


def main():
    allele = 'DQA1'
    filename = {'DQA1': 'DQA1_prot.txt', 'DQB1': 'DQB1_prot.txt', 'DRB1': 'DRB_prot.txt'}[allele]
    input_file = f'../Data/HLA-II Seq/Original/{filename}'
    output_file = f'../Data/HLA-II Seq/Processed/{allele}.csv'

    full_seq = FULL_SEQ[allele]
    seq_dict = {}
    final = []
    prot_idx = -1

    with open(input_file) as in_f:
        lines = in_f.readlines()
        for line in lines:

            if 'Prot' in line:  # 'Prot' in the beginning of each part of sequences (each seq is divided to 3 parts)
                prot_idx += 1
                if prot_idx == 3:
                    break
            if f'{allele}*' not in line:  # skip first lines
                continue

            line = line.split()
            if line[0][-1] in ['N', 'Q']:  # null allele
                continue
            hla = line[0]
            seq = ''.join(line[1:])

            two_digits = hla.split(':')[0] + ':' + hla.split(':')[1]
            if all([two_digits not in k for k in seq_dict.keys()]) or hla in seq_dict:
                curr_seq = get_curr_seq(seq, full_seq[prot_idx])
                if curr_seq is None:  # not valid. short
                    continue
                if hla not in seq_dict:  # first insertion
                    seq_dict[hla] = curr_seq
                else:  # second or third insertion
                    seq_dict[hla] += curr_seq

    for key, val in seq_dict.items():
        key_two_digits = key.split(':')[0] + ':' + key.split(':')[1]
        val_without_dot = val.replace('.', '')
        final.append([key_two_digits, val_without_dot])

    # validation
    len_SEQs = [len(i[1]) for i in final]
    if len(set(len_SEQs)) == 1:
        print('Good :)')
    else:
        print('Not good :(')

    with open(output_file, 'w') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(['HLA_NAME', 'HLA_SEQ'])
        for pair in final:
            writer.writerow(pair)

main()
