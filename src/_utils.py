# version         Beta
# date            11/25/2021
# author          Alexander-Maurice Illig
# affilation      Institute of Biotechnology, RWTH Aachen
# email           a.illig@biotec.rwth-aachen.de

from _errors import InvalidVariantError
import os

"""
Valid characters for one letter codes of amino acids. 
"""
amino_acids_olc=[
    'A','C','D','E','F',
    'G','H','I','K','L',
    'M','N','P','Q','R',
    'S','T','V','W','Y'
    ]

def is_valid_substitution(substitution: str) -> bool:
    """
    Description
    -----------
    A substitution has to follow the scheme:
    First character: (wild-type/substituted) amino acid in one letter code representation
    Last character: (introduced) amino acid in one letter code representation
    In between: position (of substitution)

    If the entered substitution does not follow this scheme (integer enclosed by two one
    letter code representations of amino acids) return False, else return True.

    Parameters
    -----------
    substitution : str
        Substitution as string: Integer enclosed by two letters representing
        the wild-type (first) and variant amino acid (last) in one letter code.

    Returns
    -------
    boolian
    """
    if not substitution[0] in amino_acids_olc:
        return False

    if not substitution[-1] in amino_acids_olc:
        return False

    try:
        int(substitution[1:-1])
    except ValueError:
        return False

    return True

def is_valid_variant(variant:str, separator=',') -> bool:
    """
    Description
    -----------
    Gets the single substitutions of the variant and checks if they follow the required scheme.

    If the entered substitution does not follow this scheme (integer enclosed by two one
    letter code representations of amino acids) return False, else return True.

    Parameters
    ----------
    variant : str
        Joined string of integers enclosed by two letters representing the wild type
        and variant amino acid in the single letter code. -> Check separator
    separator : str
        Character to split the variant to obtain the single substitutions (default=',').
    
    Returns
    -------
    boolian
    """
    for substitution in variant.split(separator):
        if not is_valid_substitution(substitution):
            return False

    return True

def get_single_substitutions(variant:str, separator=',') -> object:
    """
    Description
    -----------
    Generator that extracts and returns the single substitutions of the entered variant.

    Parameters
    ----------
    See 'is_valid_variant' for an explanation.

    Returns
    -------
    Generator object
    """
    if is_valid_variant(variant, separator):
        for substitution in variant.split(separator):
            yield substitution

    else:
        raise InvalidVariantError(variant)

def get_basename(filename: str) -> str:
    """
    Description
    -----------
    Extracts and returns the basename of the filename.

    Parameters
    ----------
    filename: str

    Returns
    -------
    str
    """
    return os.path.basename(filename).split('.')[0]