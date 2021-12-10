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

        # self._atom_list.sort(key=lambda sort_atom: sort_atom.relative_position.tolist())
        def _position_sort(lhs: Atom, rhs: Atom) -> bool:
            relative_position_lhs = lhs.relative_position
            relative_position_rhs = rhs.relative_position
            diff_norm = np.linalg.norm(relative_position_lhs - np.full((3,), 0.5)) - \
                        np.linalg.norm(relative_position_rhs - np.full((3,), 0.5))
            if diff_norm < - K_EPSILON:
                return True
            if diff_norm > K_EPSILON:
                return False
            diff_x = relative_position_lhs[0] - relative_position_rhs[0]
            if diff_x < - K_EPSILON:
                return True
            if diff_x > K_EPSILON:
                return False
            diff_y = relative_position_lhs[1] - relative_position_rhs[1]
            if diff_y < - K_EPSILON:
                return True
            if diff_y > K_EPSILON:
                return False
            diff_z = relative_position_lhs[2] - relative_position_rhs[2]
            if diff_z < - K_EPSILON:
                return True
            if diff_z > K_EPSILON:
                return False
            return lhs.atom_id < rhs.atom_id

        Atom.__lt__ = lambda this, other: _position_sort(this, other)
        self._atom_list.sort()

    def __eq__(self, other):
        for atom1, atom2 in zip(self.atom_list, other.atom_list):
            if atom1.atom_id != atom2.atom_id:
                return False
        return True

    def __hash__(self):
        atom_id_list = [atom.atom_id for atom in self._atom_list]
        the_hash = hash(tuple(atom_id_list))
        return the_hash

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


def _is_atom_smaller_symmetrically(lhs: Atom, rhs: Atom) -> bool:
    relative_position_lhs = lhs.relative_position
    relative_position_rhs = rhs.relative_position
    diff_norm = np.linalg.norm(relative_position_lhs - np.full((3,), 0.5)) - \
                np.linalg.norm(relative_position_rhs - np.full((3,), 0.5))
    if diff_norm < - K_EPSILON:
        return True
    if diff_norm > K_EPSILON:
        return False
    diff_x = relative_position_lhs[0] - relative_position_rhs[0]
    return diff_x < - K_EPSILON


def _atom_sort(lhs: Atom, rhs: Atom) -> bool:
    if _is_atom_smaller_symmetrically(lhs, rhs):
        return True
    if _is_atom_smaller_symmetrically(rhs, lhs):
        return False
    relative_position_lhs = lhs.relative_position
    relative_position_rhs = rhs.relative_position
    diff_x = relative_position_lhs[0] - relative_position_rhs[0]
    if diff_x < - K_EPSILON:
        return True
    if diff_x > K_EPSILON:
        return False
    diff_y = relative_position_lhs[1] - relative_position_rhs[1]
    if diff_y < - K_EPSILON:
        return True
    if diff_y > K_EPSILON:
        return False
    return relative_position_lhs[2] < relative_position_rhs[2] - K_EPSILON


def _is_cluster_smaller_symmetrically(lhs: Cluster, rhs: Cluster) -> bool:
    for atom1, atom2 in zip(lhs.atom_list, rhs.atom_list):
        if _is_atom_smaller_symmetrically(atom1, atom2):
            return True
        if _is_atom_smaller_symmetrically(atom2, atom1):
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
        relative_position_lhs = lhs.relative_position
        relative_position_rhs = rhs.relative_position
        diff_norm = np.linalg.norm(relative_position_lhs - np.full((3,), 0.5)) - \
                    np.linalg.norm(relative_position_rhs - np.full((3,), 0.5))
        if diff_norm < - K_EPSILON:
            return True
        if diff_norm > K_EPSILON:
            return False
        diff_x = relative_position_lhs[0] - relative_position_rhs[0]
        if diff_x < - K_EPSILON:
            return True
        if diff_x > K_EPSILON:
            return False
        diff_y = relative_position_lhs[1] - relative_position_rhs[1]
        if diff_y < - K_EPSILON:
            return True
        if diff_y > K_EPSILON:
            return False
        diff_z = relative_position_lhs[2] - relative_position_rhs[2]
        if diff_z < - K_EPSILON:
            return True
        if diff_z > K_EPSILON:
            return False
        return lhs.atom_id < rhs.atom_id

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
    vacancy_relative_position = np.zeros(3)
    vacancy_cartesian_position = np.zeros(3)

    for atom_id in atom_id_set:
        atom = copy.deepcopy(config.atom_list[atom_id])
        atom.clean_neighbors_lists()
        relative_position = atom.relative_position
        relative_position += move_distance
        relative_position -= np.floor(relative_position)
        atom.relative_position = relative_position
        if atom.atom_id == jump_pair[0]:
            vacancy_cartesian_position = atom.cartesian_position
            vacancy_relative_position = atom.relative_position
            continue
        atom_list_forward.append(atom)

    atom_list_backward = copy.deepcopy(atom_list_forward)
    for i, atom in enumerate(atom_list_backward):
        if atom.atom_id == jump_pair[1]:
            atom_list_backward[i].relative_position = vacancy_relative_position
            atom_list_backward[i].cartesian_position = vacancy_cartesian_position

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
            for atom in cluster_list[index].atom_list:
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
    for atom1 in atom_vector:
        for atom2_index in atom1.first_nearest_neighbor_list:
            first_pair_set.add(Cluster(atom1, atom_vector[atom2_index]))
        for atom2_index in atom1.second_nearest_neighbor_list:
            second_pair_set.add(Cluster(atom1, atom_vector[atom2_index]))
        for atom2_index in atom1.third_nearest_neighbor_list:
            third_pair_set.add(Cluster(atom1, atom_vector[atom2_index]))

    _get_average_parameters_mapping_from_cluster_vector_helper(list(first_pair_set), cluster_mapping)
    _get_average_parameters_mapping_from_cluster_vector_helper(list(second_pair_set), cluster_mapping)
    _get_average_parameters_mapping_from_cluster_vector_helper(list(third_pair_set), cluster_mapping)

    # # first nearest triplets
    # triplets_set: typing.Set[Cluster] = set()
    # for atom1 in atom_vector:
    #     for atom2_index in atom1.first_nearest_neighbor_list:
    #         atom2 = atom_vector[atom2_index]
    #         for atom3_index in atom2.first_nearest_neighbor_list:
    #             if atom3_index in atom1.first_nearest_neighbor_list:
    #                 triplets_set.add(Cluster(atom1, atom2, atom_vector[atom3_index]))
    # _get_average_parameters_mapping_from_cluster_vector_helper(list(triplets_set), cluster_mapping)

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
            sum_of_list = [0.0] * (len(type_set) ** len(cluster_vector[0]))
            for cluster in cluster_vector:
                cluster_type_key = ""
                for atom_index in cluster:
                    cluster_type_key += atom_vector[atom_index].elem_type
                element_wise_add_second_to_first(sum_of_list, one_hot_encode_dict[cluster_type_key])
            element_wise_divide_float_from_list(sum_of_list, float(len(cluster_vector)))
            encode_list = encode_list + sum_of_list
        result.append(encode_list)
    return tuple(result)


if __name__ == "__main__":
    config11 = cfg.read_config("../../test/test_files/forward.cfg")
    cl_mapping = get_average_cluster_parameters_mapping_symmetry(config11)
    forward, backward = get_one_hot_encoding_list_forward_and_backward_from_mapping(
        config11, (18, 23), {"Al", "Mg", "Zn"}, cl_mapping)
    for i in cl_mapping:
        for ii in i:
            ii.sort()
        print(i)
# cfg11 = get_symmetrically_sorted_configs(config11, (18, 23))
# for atom in cfg11[0].atom_list:
#     print((np.linalg.norm(atom.relative_position[1:] - np.full((2,), 0.5)),
#           np.linalg.norm(atom.relative_position - np.full((3,), 0.5))))
