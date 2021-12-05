from configtools.cfg import Atom
import copy
import typing
import numpy as np


def generate_one_hot_encode_dict_for_type(type_set: typing.Set[str]) -> typing.Dict[str, typing.List[float]]:
    sorted_type_set = sorted(type_set)
    num_singlets = len(type_set)
    encode_dict: typing.Dict[str, typing.List[float]] = dict()
    counter = 0
    for element in sorted_type_set:
        element_encode = [0.] * num_singlets
        element_encode[counter] = 1.
        encode_dict[element] = element_encode
        counter += 1

    num_pairs = len(type_set) ** 2
    counter = 0
    for element1 in sorted_type_set:
        for element2 in sorted_type_set:
            element_encode = [0.] * num_pairs
            element_encode[counter] = 1.
            encode_dict[element1 + element2] = element_encode
            counter += 1

    num_triplets = len(type_set) ** 3
    counter = 0
    for element1 in sorted_type_set:
        for element2 in sorted_type_set:
            for element3 in sorted_type_set:
                element_encode = [0.] * num_triplets
                element_encode[counter] = 1.
                encode_dict[element1 + element2 + element3] = element_encode
                counter += 1

    num_quadruplets = len(type_set) ** 4
    counter = 0
    for element1 in sorted_type_set:
        for element2 in sorted_type_set:
            for element3 in sorted_type_set:
                for element4 in sorted_type_set:
                    element_encode = [0.] * num_quadruplets
                    element_encode[counter] = 1.
                    encode_dict[element1 + element2 + element3 + element4] = element_encode
                    counter += 1

    return encode_dict


def element_wise_add_second_to_first(first_list: typing.List[float], second_list: typing.List[float]):
    if len(first_list) != len(second_list):
        raise RuntimeError("Size mismatch")
    for i in range(len(first_list)):
        first_list[i] += second_list[i]


def element_wise_divide_float_from_list(float_list: typing.List[float], divisor: float):
    for i in range(len(float_list)):
        float_list[i] /= divisor
