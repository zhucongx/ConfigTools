from .atomic_mass import *
from .constants import *

import typing
import string


def atomic_mass(term: typing.Union[int, str]) -> float:
    """
    Returns the average standard atomic weight for an element or
    the relative atomic mass for an isotope.
    Elements can be specified using atomic numbers or symbols.
    Isotopes are specified using symbols followed by -#, where # is the mass
    number (eg. 'He-3') Deuterium and tritium can alternatively be specified
    using D and T respectively.
    Values obtained from NIST reference database:
    http://www.nist.gov/pml/data/comp.cfm
    """
    if isinstance(term, (int, float)):
        term = atom_symbol_index[int(term)]
    elif type(term) is str:
        pass
    else:
        raise KeyError('Unknown input type ' + str(term))

    try:
        return atomic_mass_dict[term]
    except KeyError:
        if term in atom_symbol_index.values():
            raise ValueError('No standard weight for element ' + str(
                term) + '. Specify an isotope instead.')
        else:
            raise KeyError('Unknown element/isotope symbol: ' + str(term))


# def str2list(raw_str: str) -> list:
#     raw_list = raw_str.strip(string.whitespace).split()
#     # Remove space elements in list.
#     clean_list = [x for x in raw_list if x != ' ' and x != '']
#     return clean_list
#
#
# def line2list(line, field=' ', dtype=float):
#     "Convert text data in a line to data object list."
#     strlist = line.strip().split(field)
#     if type(dtype) != type:
#         raise TypeError('Illegal dtype.')
#     datalist = [dtype(i) for i in strlist if i != '']
#
#     return datalist
#
#
# def array2str(raw_array):
#     """
#     convert 2d array -> string
#     """
#     array_str = ''
#     for array_1d in raw_array:
#         array_str += '(%-20.16f, %-20.16f, %-20.16f)\n' % (tuple(array_1d))
#
#     return array_str
