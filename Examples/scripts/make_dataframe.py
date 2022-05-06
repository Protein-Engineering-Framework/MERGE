# version         Beta
# date            01/04/2022
# author          Alexander-Maurice Illig
# affilation      Institute of Biotechnology, RWTH Aachen
# email           a.illig@biotec.rwth-aachen.de

"""
Changes:
    - factor 0.5 for Ji term in '_encode_variant' added
    - function '_encode_wt' added
"""

"""
The included class 'CouplingsModel' has been taken from the script 'model.py' as part of the 
EVmutation module (https://github.com/debbiemarkslab/EVmutation) written by Thomas Hopf in the
labs of Debora Marks and Chris Sander at Harvard Medical School and modified (shortened).

See also:
Hopf, T. A., Ingraham, J. B., Poelwijk, F.J., Schärfe, C.P.I., Springer, M., Sander, C., & Marks, D. S. (2016).
Mutation effects predicted from sequence co-variation. Nature Biotechnology, in press.
"""

from _utils import get_single_substitutions,get_basename
from _errors import ActiveSiteError

from collections.abc import Iterable # originally imported 'from collections'
import pandas as pd
import numpy as np
import multiprocessing

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-start_pos', help='Starting position/residue number in fasta sequence.',type=int,default=1)
parser.add_argument('-params', help='PLMC output params file.')
parser.add_argument('-fitness', help='Key for fitness column.')
parser.add_argument('-csv', help='Dataframe containing the encoded variants (Col1: Variant, Col2: Fitness, Col3: X_1, ..., ColN+2: X_N')
parser.add_argument('-n_processes', help='Number of processes to be executed in parallel (default=6).', type=int, default=6)
args = parser.parse_args()

class CouplingsModel:
    """
    Class to store parameters of pairwise undirected graphical model of sequences
    and compute evolutionary couplings, sequence statistical energies, etc.
    """

    def __init__(self, filename, precision="float32", **kwargs):
        """
        Initializes the object with raw values read from binary .Jij file

        Parameters
        ----------
        filename : str
            Binary Jij file containing model parameters from plmc software
        alphabet : str
            Symbols corresponding to model states (e.g. "-ACGT").
        precision : {"float32", "float64"}, default: "float32"
            Sets if input file has single (float32) or double precision (float64)
        }
        """
        self.__read_plmc_v2(filename, precision)
        self.alphabet_map = {s: i for i, s in enumerate(self.alphabet)}

        # in non-gap mode, focus sequence is still coded with a gap character,
        # but gap is not part of model alphabet anymore; so if mapping crashes
        # that means there is a non-alphabet character in sequence array
        # and therefore there is no focus sequence.
        try:
            self.target_seq_mapped = np.array([self.alphabet_map[x] for x in self.target_seq])
            self.has_target_seq = (np.sum(self.target_seq_mapped) > 0)
        except KeyError:
            self.target_seq_mapped = np.zeros((self.L), dtype=np.int32)
            self.has_target_seq = False

    def __read_plmc_v2(self, filename, precision):
        """
        Read updated Jij file format from plmc.

        Parameters
        ----------
        filename : str
            Binary Jij file containing model parameters
        precision : {"float32", "float64"}
            Sets if input file has single or double precision

        """
        with open(filename, "rb") as f:
            # model length, number of symbols, valid/invalid sequences
            # and iterations
            self.L, self.num_symbols, self.N_valid, self.N_invalid, self.num_iter = (
                np.fromfile(f, "int32", 5)
            )

            # theta, regularization weights, and effective number of samples
            self.theta, self.lambda_h, self.lambda_J, self.lambda_group, self.N_eff = (
                np.fromfile(f, precision, 5)
            )

            # Read alphabet (make sure we get proper unicode rather than byte string)
            self.alphabet = np.fromfile(
                f, "S1", self.num_symbols
            ).astype("U1")

            # weights of individual sequences (after clustering)
            self.weights = np.fromfile(
                f, precision, self.N_valid + self.N_invalid
            )

            # target sequence and index mapping, again ensure unicode
            self._target_seq = np.fromfile(f, "S1", self.L).astype("U1")
            self.index_list = np.fromfile(f, "int32", self.L)

            # single site frequencies f_i and fields h_i
            self.f_i, = np.fromfile(
                f, dtype=(precision, (self.L, self.num_symbols)), count=1
            )

            self.h_i, = np.fromfile(
                f, dtype=(precision, (self.L, self.num_symbols)), count=1
            )

            # pair frequencies f_ij and pair couplings J_ij / J_ij
            self.f_ij = np.zeros(
                (self.L, self.L, self.num_symbols, self.num_symbols)
            )

            self.J_ij = np.zeros(
                (self.L, self.L, self.num_symbols, self.num_symbols)
            )

            for i in range(self.L - 1):
                for j in range(i + 1, self.L):
                    self.f_ij[i, j], = np.fromfile(
                        f, dtype=(precision, (self.num_symbols, self.num_symbols)),
                        count=1
                    )
                    self.f_ij[j, i] = self.f_ij[i, j].T

            for i in range(self.L - 1):
                for j in range(i + 1, self.L):
                    self.J_ij[i, j], = np.fromfile(
                        f, dtype=(precision, (self.num_symbols, self.num_symbols)),
                        count=1
                    )
                    self.J_ij[j, i] = self.J_ij[i, j].T

    @property
    def target_seq(self):
        """
        Target/Focus sequence of model used for delta_hamiltonian
        calculations (including single and double mutation matrices)
        """
        return self._target_seq

    @target_seq.setter
    def target_seq(self, sequence):
        """
        Define a new target sequence

        Parameters
        ----------
        sequence : str, or list of chars
            Define a new default sequence for relative Hamiltonian
            calculations (e.g. energy difference relative to wild-type
            sequence).
            Length of sequence must correspond to model length (self.L)
        """
        if len(sequence) != self.L:
            raise ValueError(
                "Sequence length inconsistent with model length: {} {}".format(
                    len(sequence), self.L
                )
            )

        if isinstance(sequence, str):
            sequence = list(sequence)

        self._target_seq = np.array(sequence)
        self.target_seq_mapped = np.array([self.alphabet_map[x] for x in self.target_seq])
        self.has_target_seq = True

    @property
    def index_list(self):
        """
        Target/Focus sequence of model used for delta_hamiltonian
        calculations (including single and double mutation matrices)
        """
        return self._index_list

    @index_list.setter
    def index_list(self, mapping):
        """
        Define a new number mapping for sequences

        Parameters
        ----------
        mapping: list of int
            Sequence indices of the positions in the model.
            Length of list must correspond to model length (self.L)
        """
        if len(mapping) != self.L:
            raise ValueError(
                "Mapping length inconsistent with model length: {} {}".format(
                    len(mapping), self.L
                )
            )

        self._index_list = np.array(mapping)
        self.index_map = {b: a for a, b in enumerate(self.index_list)}

    def __map(self, indices, mapping):
        """
        Applies a mapping either to a single index, or to a list of indices

        Parameters
        ----------
        indices : Iterable of items to be mapped, or single item
        mapping: Dictionary containing mapping into new space

        Returns
        -------
        Iterable, or single item
            Items mapped into new space
        """
        if ((isinstance(indices, Iterable) and not isinstance(indices, str)) or
                (isinstance(indices, str) and len(indices) > 1)):
            return np.array(
                [mapping[i] for i in indices]
            )
        else:
            return mapping[indices]

    def __4d_access(self, matrix, i=None, j=None, A_i=None, A_j=None):
        """
        Provides shortcut access to column pair properties
        (e.g. J_ij or f_ij matrices)

        Parameters
        -----------
        i : Iterable(int) or int
            Position(s) on first matrix axis
        j : Iterable(int) or int
            Position(s) on second matrix axis
        A_i : Iterable(str) or str
            Symbols corresponding to first matrix axis
        A_j : Iterable(str) or str
            Symbols corresponding to second matrix axis

        Returns
        -------
        np.array
            4D matrix "matrix" sliced according to values i, j, A_i and A_j
        """
        i = self.__map(i, self.index_map) if i is not None else _SLICE
        j = self.__map(j, self.index_map) if j is not None else _SLICE
        A_i = self.__map(A_i, self.alphabet_map) if A_i is not None else _SLICE
        A_j = self.__map(A_j, self.alphabet_map) if A_j is not None else _SLICE
        return matrix[i, j, A_i, A_j]

    def __2d_access(self, matrix, i=None, A_i=None):
        """
        Provides shortcut access to single-column properties
        (e.g. f_i or h_i matrices)

        Parameters
        -----------
        i : Iterable(int) or int
            Position(s) on first matrix axis
        A_i : Iterable(str) or str
            Symbols corresponding to first matrix axis

        Returns
        -------
        np.array
            2D matrix "matrix" sliced according to values i and A_i
        """
        i = self.__map(i, self.index_map) if i is not None else _SLICE
        A_i = self.__map(A_i, self.alphabet_map) if A_i is not None else _SLICE
        return matrix[i, A_i]

    def Jij(self, i=None, j=None, A_i=None, A_j=None):
        """
        Quick access to J_ij matrix with automatic index mapping.
        See __4d_access for explanation of parameters.
        """
        return self.__4d_access(self.J_ij, i, j, A_i, A_j)

    def hi(self, i=None, A_i=None):
        """
        Quick access to h_i matrix with automatic index mapping.
        See __2d_access for explanation of parameters.
        """
        return self.__2d_access(self.h_i, i, A_i)


class Encode(CouplingsModel):
    """
    Class for performing the 'DCA encoding'.

    Attributes
    ----------
    starting_position: int
        Number of leading residue of the fasta sequence used for model construction.
    params_file: str
        Binary parameter file outputed by PLMC.
    """

    def __init__(self,starting_position: int ,params_file: str):
        self.starting_position=starting_position
        super().__init__(params_file) # inherit functions and variables from class 'CouplingsModel'

    def _get_position_internal(self,position: int):
        """
        Description
        -----------
        Returns the "internal position" of an amino acid, e.g., D19V is the desired substitution,
        but the fasta sequence starts from residue 3, .i.e, the first two residues are "missing".
        The DCA model will then recognize D19 as D17. In order to avoid wrong assignments,
        it is inevitable to calculate the "internal position" 'i'.

        Parameters
        ----------
        position : int
            Position of interest
        
        Returns
        -------
        i : int
            "Internal position" that may differ due to different starting residue.
        None
            If the requested position is not an active site.
        """
        offset=self.starting_position-1
        i=position-offset
        if i in self.index_list:
            return i
        else:
            return None

    def Ji(self,i: int,A_i: str,sequence: np.ndarray) -> float:
        """
        Description
        -----------
        Caluclates the sum of all site-site interaction terms when site 'i' is occupied with amino acid 'A_i'.

        Parameters
        ----------
        i : int
            "Internal position" see '_get_position_internal' for an explanation.
        A_i : str
            Introduced amino acid at 'i' in one letter code representation.
        sequence: np.ndarray
            Sequence of the variant as numpy array.

        Returns
        -------
        Ji : float
            Sum of all site-site interaction terms acting on position 'i' when occupied with 'A_i'.
        """
        Ji=0.0
        for j,A_j in zip(self.index_list,sequence):
            Ji+=self.Jij(i=i,A_i=A_i,j=j,A_j=A_j)
        return Ji

    @staticmethod
    def _unpack_substitution(substitution: str) -> tuple:
        """
        Description
        -----------
        Turns string representation of variant into tuple.

        Parameters
        ----------
        substitution : str
            Substitution as string: Integer enclosed by two letters representing
            the wild-type (first) and variant amino acid (last) in one letter code.

        Returns
        -------
        substitution : tuple
            (wild-type amino acid, position, variant amino acid)
        """
        return (substitution[0],int(substitution[1:-1]),substitution[-1])

    def _encode_variant(self,variant: str,separator=',') -> np.ndarray:
        """
        Description
        -----------
        Encodes the variant using its "DCA representation".

        Parameters
        ----------
        variant : str
            Joined string of integers enclosed by two letters representing the wild-type
            and variant amino acid in the single letter code. -> Check separator
        separator : str
            Character to split the variant to obtain the single substitutions (default=',').
        
        Returns
        -------
        X_var : np.ndarray
            Encoded sequence of the variant.
        """
        sequence=self.target_seq.copy()
        for substitution in get_single_substitutions(variant,separator):
            wild_type_aa,position,A_i=self._unpack_substitution(substitution)
         
            i=self._get_position_internal(position)
            if not i:
                raise ActiveSiteError(position,variant)

            i_mapped=self.index_map[i]
            sequence[i_mapped]=A_i

        X_var=np.zeros(sequence.size,dtype=float)
        for idx,(i,A_i) in enumerate(zip(self.index_list,sequence)):                
            X_var[idx]=self.hi(i,A_i) + 0.5*self.Ji(i,A_i,sequence)
        return X_var

    def _encode_wt(self) -> np.ndarray:
        """
        Description
        -----------
        Encodes the wild-type using its "DCA representation".
        
        Returns
        -------
        X_wt : np.ndarray
            Encoded sequence of the wild-type.
        """
        X_wt=np.zeros(self.target_seq.size,dtype=float)
        for idx,(i,A_i) in enumerate(zip(self.index_list,self.target_seq)):
            X_wt[idx]=self.hi(i,A_i) + 0.5*self.Ji(i,A_i,self.target_seq)
        return X_wt


def _get_data(variants: list,fitnesses: list,dca_encode: object,data: list) -> list:
    """
    Description
    -----------
    Get the variant name, the associated fitness value, and its ("DCA"-)encoded sequence.

    Parameters
    ----------
    variants : list
        List of strings containing the variants to be encoded.
    fitnesses : list
        List of floats (1d) containing the fitness values associated to the variants.
    dca_encode : object
        Initialized 'Encode' class object.
    data : manager.list()
        Manager.list() object to store the output of multiple processors. 

    Returns
    -------
    data : manager.list()
        Filled list with variant names, fitnesses, and encoded sequence.
    """
    for variant,fitness in zip(variants,fitnesses):
        try:
            data.append([variant,dca_encode._encode_variant(variant),fitness])
            
        except ActiveSiteError:
            pass 

def get_data(fitness_key: str,csv_file: str,dca_encode: object,n_processes: int):
    """
    Description
    -----------
    This function allows to generate the encoded sequences based on the variants
    given in 'csv_file' in a parallel manner.
    
    Parameters
    ----------
    fitness_key : str
        Name of column containing the fitness values.
    csv_file : str
        Name of the csv file containing variant names and associated fitness values.
    dca_encode : object
        Initialized 'Encode' class object.
    n_processes : int
        Number of processes to be used for parallel execution (default=6).

    Returns
    -------
    data : np.ndarray
        Filled numpy array including variant names, fitnesses, and encoded sequences.
    """

    df=pd.read_csv(csv_file,sep=';',comment='#')

    fitnesses=df[fitness_key].to_numpy()
    variants=df['variant'].to_numpy()

    idxs_nan=np.array([i for i,b in enumerate(np.isnan(fitnesses)) if b]) # find NaNs
    if idxs_nan.size>0: # remove NaNs if presented
        print('NaNs are:', idxs_nan)
        fitnesses=np.delete(fitnesses,idxs_nan)
        variants=np.delete(variants,idxs_nan)

    fitnesses_split=np.array_split(fitnesses,n_processes)
    variants_split=np.array_split(variants,n_processes)

    manager=multiprocessing.Manager()
    data=manager.list()

    processes=[]
    for variants,fitnesses in zip(variants_split,fitnesses_split):
        p=multiprocessing.Process(target=_get_data, args=[variants,fitnesses,dca_encode,data])
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return np.array(data,dtype=object)
    
def generate_dataframe(data: np.ndarray, csv_file: str):
    """
    Description
    -----------
    Takes 'csv_file' and generates new csv file containing the variants, fitnesses, and encoded sequences.
    Parameters
    ----------
    data : np.ndarray
        Filled numpy array including variant names, fitnesses, and encoded sequence.
    csv_file : str
        Name of the csv file containing variant names and associated fitness values.
    """
    variants,X,y=np.array(data,dtype=object).T # Can cause error if data.size==0 ?!
    X=np.stack(X)

    df_dca=pd.DataFrame()
    df_dca.insert(0,'variant',variants)
    df_dca.insert(1,'y',y)

    for i in range(X.shape[1]):
        df_dca.insert(i+2,'X%d'%(i+1),X[:,i])

    filename='%s_encoded.csv'%(get_basename(csv_file))
    df_dca.to_csv(filename,sep=';',index=False)


if __name__=='__main__':
    dca_encode=Encode(args.start_pos,args.params)
    data=get_data(args.fitness,args.csv,dca_encode,args.n_processes)
    generate_dataframe(data,args.csv)

    np.save('%s_wt_encoded.npy'%(get_basename(args.csv)),dca_encode._encode_wt())