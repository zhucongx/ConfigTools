from configtools.cfg import Atom
import copy
import typing
import numpy as np


def generate_one_hot_encode_dict_for_type(type_set: typing.Set[str]) -> typing.Dict[str, typing.List[float]]:
    sorted_type_list = sorted(list((type_set)))
    num_elements = len(type_set)
    num_singlets = num_elements
    encode_dict: typing.Dict[str, typing.List[float]] = dict()
    counter = 0
    for element in sorted_type_list:
        element_encode = [0.] * num_singlets
        element_encode[counter] = 1.
        encode_dict[element] = element_encode
        counter += 1

    num_pairs = num_elements ** 2
    counter = 0
    for element1 in sorted_type_list:
        for element2 in sorted_type_list:
            element_encode = [0.] * num_pairs
            element_encode[counter] = 1.
            encode_dict[element1 + element2] = element_encode
            counter += 1

    num_pairs_symmetry = int((num_elements + 1) * num_elements / 2)
    counter = 0
    for i in range(len(sorted_type_list)):
        for j in range(i, len(sorted_type_list)):
            element_encode = [0.] * num_pairs_symmetry
            element_encode[counter] = 1.
            encode_dict['-'.join([sorted_type_list[i], sorted_type_list[j]])] = element_encode
            counter += 1

    # num_triplets = num_elements ** 3
    # counter = 0
    # for element1 in sorted_type_list:
    #     for element2 in sorted_type_list:
    #         for element3 in sorted_type_list:
    #             element_encode = [0.] * num_triplets
    #             element_encode[counter] = 1.
    #             encode_dict[element1 + element2 + element3] = element_encode
    #             counter += 1

    # num_triplets_symmetry = int(
    #     np.math.factorial(num_elements + 3 - 1) / np.math.factorial(num_elements - 1) / np.math.factorial(3))
    # counter = 0
    # pairs_set = set()
    # for element1 in sorted_type_set:
    #     for element2 in sorted_type_set:
    #         for element3 in sorted_type_set:
    #             element_list = [element1, element2, element3]
    #             element_list.sort()
    #             element_symbol = '-'.join(element_list)
    #             element_encode = [0.] * num_triplets_symmetry
    #             element_encode[counter] = 1.
    #             if element_symbol not in pairs_set:
    #                 counter += 1
    #             else:
    #                 continue
    #             pairs_set.add(element_symbol)
    #             encode_dict[element_symbol] = element_encode

    return encode_dict


def element_wise_add_second_to_first(first_list: typing.List[float], second_list: typing.List[float]):
    if len(first_list) != len(second_list):
        raise RuntimeError("Size mismatch")
    for i in range(len(first_list)):
        first_list[i] += second_list[i]


def element_wise_divide_float_from_list(float_list: typing.List[float], divisor: float):
    for i in range(len(float_list)):
        float_list[i] /= divisor


if __name__ == "__main__":
    a = generate_one_hot_encode_dict_for_type({"Al", "Mg", "Zn"})
    for aa in a:
        print(aa, a[aa])
