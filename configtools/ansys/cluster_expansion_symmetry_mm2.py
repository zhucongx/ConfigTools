import typing
import bisect
import copy
import logging
from collections import OrderedDict
import numpy as np

from configtools import cfg
from configtools.ansys.cluster import *

K_EPSILON = 1e-8


class Cluster(object):
    def __init__(self, *atoms: Atom):
        self._atom_list: typing.List[Atom] = list()
        for insert_atom in atoms:
            if insert_atom.elem_type == "X":
                print("type X..!!!")
            self._atom_list.append(copy.deepcopy(insert_atom))

        # self._atom_list.sort(key=lambda sort_atom: sort_atom.fractional_position.tolist())
        def _position_sort(lhs: Atom, rhs: Atom) -> bool:
            fractional_position_lhs = lhs.fractional_position
            fractional_position_rhs = rhs.fractional_position
            diff_norm = np.linalg.norm(fractional_position_lhs - np.full((3,), 0.5)) - \
                        np.linalg.norm(fractional_position_rhs - np.full((3,), 0.5))
            if diff_norm < - K_EPSILON:
                return True
            if diff_norm > K_EPSILON:
                return False
            diff_x = fractional_position_lhs[0] - fractional_position_rhs[0]
            if diff_x < - K_EPSILON:
                return True
            if diff_x > K_EPSILON:
                return False
            diff_y = fractional_position_lhs[1] - fractional_position_rhs[1]
            if diff_y < - K_EPSILON:
                return True
            if diff_y > K_EPSILON:
                return False
            diff_z = fractional_position_lhs[2] - fractional_position_rhs[2]
            return diff_z < - K_EPSILON

        Atom.__lt__ = lambda this, other: _position_sort(this, other)
        self._atom_list.sort()

        self._symmetry_label = False
        if len(self._atom_list) == 2:
            if not (_is_atom_smaller_symmetrically(self._atom_list[0], self._atom_list[1])) and not (
                    _is_atom_smaller_symmetrically(self._atom_list[1], self._atom_list[0])):
                self._symmetry_label = True

    @property
    def atom_list(self) -> typing.List[Atom]:
        return self._atom_list

    @property
    def type_key(self) -> str:
        key = ""
        for atom in self._atom_list:
            key += atom.elem_type
        return key

    @property
    def size(self) -> int:
        return len(self._atom_list)

    @property
    def symmetry_label(self):
        return self._symmetry_label


def _is_atom_smaller_symmetrically(lhs: Atom, rhs: Atom) -> bool:
    fractional_position_lhs = lhs.fractional_position
    fractional_position_rhs = rhs.fractional_position
    diff_norm = np.linalg.norm(fractional_position_lhs - np.full((3,), 0.5)) - \
                np.linalg.norm(fractional_position_rhs - np.full((3,), 0.5))
    if diff_norm < - K_EPSILON:
        return True
    if diff_norm > K_EPSILON:
        return False
    diff_x = fractional_position_lhs[0] - fractional_position_rhs[0]
    return diff_x < - K_EPSILON


def _is_cluster_smaller_symmetrically(lhs: Cluster, rhs: Cluster) -> bool:
    if lhs.size == 1:
        return _is_atom_smaller_symmetrically(lhs.atom_list[0], rhs.atom_list[0])
    if lhs.size == 2:
        if _is_atom_smaller_symmetrically(lhs.atom_list[0], rhs.atom_list[0]):
            return True
        if _is_atom_smaller_symmetrically(rhs.atom_list[0], lhs.atom_list[0]):
            return False
        if _is_atom_smaller_symmetrically(lhs.atom_list[1], rhs.atom_list[1]):
            return True
        if _is_atom_smaller_symmetrically(rhs.atom_list[1], lhs.atom_list[1]):
            return False
    # if it reaches here, it means that the clusters are same symmetrically.Returns false.
    return False


def _rotate_atom_vector_and_sort_helper(atom_list: typing.List[Atom], reference_config: cfg.Config,
                                        jump_pair: typing.Tuple[int, int]) -> typing.List[Atom]:
    """
    Rotate the atom list in place, reference config does not change
    Parameters
    ----------
    atom_list
    reference_config
    jump_pair

    Returns
    -------

    """
    cfg.rotate_atom_vector(atom_list, cfg.get_pair_rotation_matrix(reference_config, jump_pair))

    def _position_sort(lhs: Atom, rhs: Atom) -> bool:
        fractional_position_lhs = lhs.fractional_position
        fractional_position_rhs = rhs.fractional_position
        diff_norm = np.linalg.norm(fractional_position_lhs - np.full((3,), 0.5)) - \
                    np.linalg.norm(fractional_position_rhs - np.full((3,), 0.5))
        if diff_norm < - K_EPSILON:
            return True
        if diff_norm > K_EPSILON:
            return False
        diff_x = fractional_position_lhs[0] - fractional_position_rhs[0]
        if diff_x < - K_EPSILON:
            return True
        if diff_x > K_EPSILON:
            return False
        diff_y = fractional_position_lhs[1] - fractional_position_rhs[1]
        if diff_y < - K_EPSILON:
            return True
        if diff_y > K_EPSILON:
            return False
        diff_z = fractional_position_lhs[2] - fractional_position_rhs[2]
        return diff_z < - K_EPSILON

    Atom.__lt__ = lambda self, other: _position_sort(self, other)
    atom_list.sort()

    for i, atom in enumerate(atom_list):
        atom_list[i].atom_id = i
    out_config = cfg.Config(reference_config.basis, atom_list)
    out_config.update_neighbors()
    return out_config.atom_list


def _get_symmetrically_sorted_atom_vectors(config: cfg.Config, jump_pair: typing.Tuple[int, int]) -> \
        typing.Tuple[typing.List[Atom], typing.List[Atom]]:
    """
    Returns forward and backward sorted atom lists
    Parameters
    ----------
    config
    jump_pair

    Returns
    -------

    """
    atom_id_set = cfg.get_neighbors_set_of_jump_pair(config, jump_pair)
    move_distance = np.full((3,), 0.5) - cfg.get_pair_center(config, jump_pair)
    atom_list_forward: typing.List[Atom] = list()

    for atom_id in atom_id_set:
        atom = copy.deepcopy(config.atom_list[atom_id])
        atom.clean_neighbors_lists()
        fractional_position = atom.fractional_position
        fractional_position += move_distance
        fractional_position -= np.floor(fractional_position)
        atom.fractional_position = fractional_position
        if atom.atom_id == jump_pair[0]:
            continue
        if atom.atom_id == jump_pair[1]:
            continue
        atom_list_forward.append(atom)

    atom_list_backward = copy.deepcopy(atom_list_forward)

    return _rotate_atom_vector_and_sort_helper(atom_list_forward, config, jump_pair), \
           _rotate_atom_vector_and_sort_helper(atom_list_backward, config, jump_pair[::-1])


def get_symmetrically_sorted_configs(config: cfg.Config, jump_pair: typing.Tuple[int, int]) -> \
        typing.Tuple[cfg.Config, cfg.Config]:
    sorted_vectors = _get_symmetrically_sorted_atom_vectors(config, jump_pair)
    return cfg.Config(config.basis, sorted_vectors[0]), cfg.Config(config.basis, sorted_vectors[1])


def _get_average_parameters_mapping_from_cluster_vector_helper(
        cluster_list: typing.List[Cluster],
        cluster_mapping: typing.List[typing.List[typing.List[int]]]):
    """
    Cluster_mapping will be modified

    Parameters
    ----------
    cluster_list
    cluster_mapping

    Returns
    -------

    """
    # sort clusters
    Cluster.__lt__ = lambda self, other: _is_cluster_smaller_symmetrically(self, other)
    cluster_list.sort()

    lower_bound = 0
    while True:
        # print("start index", lower_bound)
        upper_bound = bisect.bisect_right(cluster_list, cluster_list[lower_bound])
        # print("end index", upper_bound - 1)
        cluster_index_vector: typing.List[typing.List[int]] = list()
        for index in range(lower_bound, upper_bound):
            cluster_index: typing.List[int] = list()
            cluster = cluster_list[index]
            if cluster.symmetry_label:
                cluster_index.append(-1)
            for atom in cluster.atom_list:
                cluster_index.append(atom.atom_id)
            cluster_index_vector.append(cluster_index)
        cluster_mapping.append(cluster_index_vector)

        lower_bound = upper_bound
        if lower_bound == len(cluster_list):
            break


def get_average_cluster_parameters_mapping_symmetry(config: cfg.Config) -> typing.List[typing.List[typing.List[int]]]:
    vacancy_index = cfg.get_vacancy_index(config)
    neighbor_index = config.atom_list[vacancy_index].first_nearest_neighbor_list[0]

    atom_vector = _get_symmetrically_sorted_atom_vectors(config, (vacancy_index, neighbor_index))[0]

    cluster_mapping: typing.List[typing.List[typing.List[int]]] = list()
    # singlets
    singlet_vector: typing.List[Cluster] = list()
    for atom in atom_vector:
        singlet_vector.append(Cluster(atom))
    _get_average_parameters_mapping_from_cluster_vector_helper(singlet_vector, cluster_mapping)
    # pairs
    first_pair_set: typing.Set[Cluster] = set()
    second_pair_set: typing.Set[Cluster] = set()
    third_pair_set: typing.Set[Cluster] = set()
    for index1 in range(len(atom_vector)):
        for index2 in range(index1):
            t = cfg.get_bond_length_type_between(atom_vector[index1], atom_vector[index2])
            if t == 1:
                first_pair_set.add(Cluster(atom_vector[index1], atom_vector[index2]))
            elif t == 2:
                second_pair_set.add(Cluster(atom_vector[index1], atom_vector[index2]))
            elif t == 3:
                third_pair_set.add(Cluster(atom_vector[index1], atom_vector[index2]))
    _get_average_parameters_mapping_from_cluster_vector_helper(list(first_pair_set), cluster_mapping)
    _get_average_parameters_mapping_from_cluster_vector_helper(list(second_pair_set), cluster_mapping)
    _get_average_parameters_mapping_from_cluster_vector_helper(list(third_pair_set), cluster_mapping)

    return cluster_mapping


# def get_average_cluster_parameters_forward_and_backward_from_map(
#         config: cfg.Config, jump_pair: typing.Tuple[int, int],
#         type_category_map: typing.Dict[str, float],
#         cluster_mapping: typing.List[typing.List[typing.List[int]]]) -> \
#         typing.Tuple[typing.List[float], typing.List[float]]:
#     result: typing.List[typing.List[float]] = list()
#     atom_vectors = _get_symmetrically_sorted_atom_vectors(config, jump_pair)
#     for atom_vector in atom_vectors:
#         encode_list: typing.List[float] = list()
#         encode_list.append(1.0)
#         for cluster_vector in cluster_mapping:
#             sum_of_functional = 0.0
#             for cluster in cluster_vector:
#                 cumulative_product = 1.0
#                 for atom_index in cluster:
#                     cumulative_product *= type_category_map[atom_vector[atom_index].elem_type]
#                 sum_of_functional += cumulative_product
#             encode_list.append(sum_of_functional / len(cluster_vector))
#         result.append(encode_list)
#     return tuple(result)


def get_one_hot_encoding_list_forward_and_backward_from_mapping(
        config: cfg.Config,
        jump_pair: typing.Tuple[int, int],
        type_set: typing.Set[str],
        cluster_mapping: typing.List[typing.List[typing.List[int]]]) -> \
        typing.Tuple[typing.List[float], typing.List[float]]:
    one_hot_encode_dict = generate_one_hot_encode_dict_for_type(type_set)
    result: typing.List[typing.List[float]] = list()
    atom_vectors = _get_symmetrically_sorted_atom_vectors(config, jump_pair)
    for atom_vector in atom_vectors:
        encode_list: typing.List[float] = list()
        for cluster_vector in cluster_mapping:
            if cluster_vector[0][0] == -1:
                num_elements = len(type_set)
                select_num = len(cluster_vector[0]) - 1
                sum_of_list = [0.0] * int(np.math.factorial(
                    num_elements + select_num - 1) / np.math.factorial(
                    num_elements - 1) / np.math.factorial(select_num))
            else:
                sum_of_list = [0.0] * (len(type_set) ** len(cluster_vector[0]))
            for cluster in cluster_vector:
                cluster_type_keys = []
                if cluster[0] == -1:
                    for atom_index in cluster[1:]:
                        cluster_type_keys.append(atom_vector[atom_index].elem_type)
                    cluster_type_keys.sort()
                    cluster_type_key = "-".join(cluster_type_keys)
                else:
                    for atom_index in cluster:
                        cluster_type_keys.append(atom_vector[atom_index].elem_type)
                    cluster_type_key = "".join(cluster_type_keys)
                element_wise_add_second_to_first(sum_of_list, one_hot_encode_dict[cluster_type_key])
            element_wise_divide_float_from_list(sum_of_list, float(len(cluster_vector)))
            encode_list = encode_list + sum_of_list
        result.append(encode_list)
    return tuple(result)


if __name__ == "__main__":
    configm = cfg.read_config("../../test/test_files/mapping.cfg")
    configf = cfg.read_config("../../test/test_files/forward.cfg")
    configb = cfg.read_config("../../test/test_files/backward.cfg")

    cl_mapping = get_average_cluster_parameters_mapping_symmetry(configm)
    cl_mapping1 = get_average_cluster_parameters_mapping_symmetry(configf)
    cl_mapping2 = get_average_cluster_parameters_mapping_symmetry(configb)
    # for i in cl_mapping2:
    #     print(i)
    # for i in cl_mapping2:
    #     print(i)
    forward1, backward1 = get_one_hot_encoding_list_forward_and_backward_from_mapping(
        configf, (18, 23), {"Al", "Mg", "Zn"}, cl_mapping1)
    forward2, backward2 = get_one_hot_encoding_list_forward_and_backward_from_mapping(
        configb, (23, 18), {"Al", "Mg", "Zn"}, cl_mapping2)
    print(forward1)
    print(backward1)
    print(forward2)
    print(backward2)
    # forward2, backward2 = get_one_hot_encoding_list_forward_and_backward_from_mapping(
    #     configb, (23, 18), {"Al", "Mg", "Zn"}, cl_mapping2)
    # print((np.array(backward1) - np.array(forward1)).tolist())
    # print((np.array(forward2) - np.array(backward2)).tolist())
    # index_map = {
    #     -1: -1,
    #
    #     0: 11,
    #     1: 8,
    #     2: 10,
    #     3: 9,
    #
    #     4: 12,
    #     5: 13,
    #
    #     6: 6,
    #     7: 7,
    #
    #     8: 17,
    #     9: 14,
    #     10: 16,
    #     11: 15,
    #
    #     12: 5,
    #     13: 2,
    #     14: 4,
    #     15: 3,
    #
    #     16: 18,
    #     17: 1,
    # }
    # for i in cl_mapping1:
    #     lli = []
    #     for j in i:
    #         li = []
    #         for k in j:
    #             li.append(index_map[k])
    #         lli.append(li)
    #     print(lli)
# cfg11 = get_symmetrically_sorted_configs(config11, (18, 23))
# for atom in cfg11[0].atom_list:
#     print((np.linalg.norm(atom.fractional_position[1:] - np.full((2,), 0.5)),
#           np.linalg.norm(atom.fractional_position - np.full((3,), 0.5))))
