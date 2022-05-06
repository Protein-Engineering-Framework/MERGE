"""
Author: Alexander-Maurice Illig
Date: 2020/10/25
Institute: Biotechnology
University: RWTH Aachen
"""

# Import the necessary packages 
import argparse
import numpy as np
from tqdm import tqdm
from Bio import AlignIO


# Load parser
parser = argparse.ArgumentParser()
parser.add_argument('-sto', help="Filename of the alignment in stockholm format to be converted into fasta format.")
parser.add_argument('-inter_gap', help="Fraction to delete all positions with more than 'inter_gap' * 100 %% gaps (columnar trimming). | default=0.3",default=0.3, type=float)
parser.add_argument('-intra_gap', help="Fraction to delete all sequences with more than 'intra_gap' * 100 %% gaps after being columnar trimmed (line trimming). | default=0.5",default=0.5, type=float)
args = parser.parse_args()


def convert_sto2a2m(sto_file:str, inter_gap:float, intra_gap:float):
    # Generate the a2m output filename
    a2m_file="%s.a2m"%(sto_file.split(".sto")[0])


    # Load the stockholm alignment
    print('Loading MSA in stockholm format...')
    sto_alignment=AlignIO.read(sto_file, 'stockholm')
    print('Trimming MSA...')
    # Save this 'raw' multiple sequence alignment as numpy array 
    raw_msa=[]
    for record in tqdm(sto_alignment):
        raw_msa.append(np.array(record.seq))
    raw_msa=np.array(raw_msa)


    # 1st processing step
    # Delete all positions, where WT has a gap to obtain the 'trimmed' MSA
    ungap_pos=np.where(raw_msa[0]=="-")
    msa_trimmed=np.array([np.delete(seq, ungap_pos) for seq in raw_msa])


    # 2nd processing step
    # Remove ("lower") all positions with more than 'inter_gap'*100 % gaps (columnar trimming)
    count_gaps=np.count_nonzero(msa_trimmed=='-', axis=0)/msa_trimmed.shape[0]
    lower=[idx for idx, count in enumerate(count_gaps) if count > inter_gap]     
    msa_trimmed_T=msa_trimmed.T
    for idx in lower:
        msa_trimmed_T[idx]=np.char.lower(msa_trimmed_T[idx])    
        # replace all columns that are "removed" due to high gap content and have an "-" element by "." 
        msa_trimmed_T[idx]=np.where(msa_trimmed_T[idx] == '-', '.' , msa_trimmed_T[idx])
    msa_trimmed_inter_gap=msa_trimmed_T.T


    # 3rd processing step
    # Remove all sequences with more than 'intra_gap'*100 % gaps (line trimming)
    target_len=len(msa_trimmed_inter_gap[0])
    gap_content=(np.count_nonzero(msa_trimmed_inter_gap=="-",axis=1) + np.count_nonzero(msa_trimmed_inter_gap==".",axis=1))/target_len
    delete=np.where(gap_content > intra_gap)[0]
    msa_final=np.delete(msa_trimmed_inter_gap,delete,axis=0)
    seqs_cls=[seq_cls for idx, seq_cls in enumerate(sto_alignment) if not idx in delete]
    chunkSize=60
    with open(a2m_file, 'w') as f:
        for seq, seq_cls in zip(msa_final, seqs_cls):
            f.write('>' + seq_cls.id + '\n')
            for chunk in [seq[x:x+chunkSize] for x in range(0, len(seq), chunkSize)]:
                f.write("".join(chunk) + '\n')



    # Get number of sequences and active sites in the aligment
    n_seqs=msa_final.shape[0]
    n_sites=sum(1 for char in msa_final[0] if char.isupper())
    print('Generated trimmed MSA %s in A2M format:'%a2m_file)

    return n_seqs,n_sites,target_len



if __name__=="__main__":
    n_seqs,n_active_sites,n_sites=convert_sto2a2m(args.sto, args.inter_gap, args.intra_gap)

    print("N_seqs: %d"%(n_seqs))
    print("N_active_sites: %d (out of %d sites)"%(n_active_sites,n_sites))
    print("-le: %.1f"%(0.2*(n_active_sites-1)))