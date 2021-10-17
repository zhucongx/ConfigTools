import typing
import numpy as np

import cfg
from ansys.cluster import *
from collections import OrderedDict
import bisect
import copy

K_EPSILON = 1e-8


def _is_atom_smaller(lhs: Atom, rhs: Atom) -> bool:
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
        if _is_atom_smaller(atom1, atom2):
            return True
        if _is_atom_smaller(atom2, atom1):
            return False
    # if it reaches here, it means that the clusters are same symmetrically.Returns false.
    return False


def _get_sorted_atom_vector(config: cfg.Config) -> typing.List[Atom]:
    move_distance = -config.atom_list[cfg.get_vacancy_index(config)].relative_position
    atom_list: typing.List[Atom] = list()
    for atom_id in range(config.number_atoms):
        atom = copy.deepcopy(config.atom_list[atom_id])
        atom.clean_neighbors_lists()
        relative_position = atom.relative_position
        relative_position += move_distance
        relative_position -= np.floor(relative_position)
        atom.relative_position = relative_position
        atom_list.append(atom)
    Atom.__lt__ = lambda self, other: _is_atom_smaller(self, other)
    atom_list.sort()
    for i, atom in enumerate(atom_list):
        atom_list[i].atom_id = i
    out_config = cfg.Config(config.basis, atom_list)
    out_config.update_neighbors()
    return out_config.atom_list


def _get_average_parameters_mapping_from_cluster_vector_helper(
        cluster_list: typing.List[Cluster],
        cluster_mapping: typing.List[typing.List[typing.List[int]]]) -> None:
    # sort clusters
    # Cluster.__lt__ = lambda self, other: _is_cluster_smaller_symmetrically(self, other)
    # cluster_list.sort()
    cluster_index_vector: typing.List[typing.List[int]] = list()
    for cluster in cluster_list:
        cluster_index: typing.List[int] = list()
        for atom in cluster.atom_list:
            cluster_index.append(atom.atom_id)
        cluster_index_vector.append(cluster_index)
    cluster_mapping.append(cluster_index_vector)


def get_average_cluster_parameters_mapping_periodic(config: cfg.Config) -> typing.List[typing.List[typing.List[int]]]:
    atom_list = _get_sorted_atom_vector(config)
    cluster_mapping: typing.List[typing.List[typing.List[int]]] = list()
    # singlets
    singlet_vector: typing.List[Cluster] = list()
    for atom in atom_list:
        if atom.elem_type == "X":
            continue
        singlet_vector.append(Cluster(atom))
    _get_average_parameters_mapping_from_cluster_vector_helper(singlet_vector, cluster_mapping)

    # first nearest pairs
    first_pair_set: typing.Set[Cluster] = set()
    second_pair_set: typing.Set[Cluster] = set()
    third_pair_set: typing.Set[Cluster] = set()

    for atom1 in atom_list:
        if atom1.elem_type == "X":
            continue
        for atom2_index in atom1.first_nearest_neighbor_list:
            atom2 = atom_list[atom2_index]
            if atom2.elem_type == "X":
                continue
            first_pair_set.add(Cluster(atom1, atom2))
        for atom2_index in atom1.second_nearest_neighbor_list:
            atom2 = atom_list[atom2_index]
            if atom2.elem_type == "X":
                continue
            second_pair_set.add(Cluster(atom1, atom2))
        for atom2_index in atom1.third_nearest_neighbor_list:
            atom2 = atom_list[atom2_index]
            if atom2.elem_type == "X":
                continue
            third_pair_set.add(Cluster(atom1, atom2))
    _get_average_parameters_mapping_from_cluster_vector_helper(list(first_pair_set), cluster_mapping)
    _get_average_parameters_mapping_from_cluster_vector_helper(list(second_pair_set), cluster_mapping)
    _get_average_parameters_mapping_from_cluster_vector_helper(list(third_pair_set), cluster_mapping)

    # triplets
    # 1-1-1
    first_first_first_triplets_set: typing.Set[Cluster] = set()
    # 1-1-2
    first_first_second_triplets_set: typing.Set[Cluster] = set()
    # 1-1-3
    first_first_third_triplets_set: typing.Set[Cluster] = set()
    # 1-2-2
    # not exist
    # 1-2-3
    first_second_third_triplets_set: typing.Set[Cluster] = set()
    # 1-3-3
    first_third_third_triplets_set: typing.Set[Cluster] = set()
    # 2-2-2
    # not exist
    # 2-2-3
    # not exist
    # 2-3-3
    second_third_third_triplets_set: typing.Set[Cluster] = set()
    # 3-3-3
    third_third_third_triplets_set: typing.Set[Cluster] = set()
    for atom1 in atom_list:
        if atom1.elem_type == "X":
            continue
        for atom2_index in atom1.first_nearest_neighbor_list:
            atom2 = atom_list[atom2_index]
            if atom2.elem_type == "X":
                continue
            for atom3_index in atom2.first_nearest_neighbor_list:
                atom3 = atom_list[atom3_index]
                if atom3.elem_type == "X":
                    continue
                if atom3_index in atom1.first_nearest_neighbor_list:
                    first_first_first_triplets_set.add(Cluster(atom1, atom2, atom3))
                if atom3_index in atom1.second_nearest_neighbor_list:
                    first_first_second_triplets_set.add(Cluster(atom1, atom2, atom3))
                if atom3_index in atom1.third_nearest_neighbor_list:
                    first_first_third_triplets_set.add(Cluster(atom1, atom2, atom3))
            for atom3_index in atom2.second_nearest_neighbor_list:
                atom3 = atom_list[atom3_index]
                if atom3.elem_type == "X":
                    continue
                if atom3_index in atom1.third_nearest_neighbor_list:
                    first_second_third_triplets_set.add(Cluster(atom1, atom2, atom3))
            for atom3_index in atom2.third_nearest_neighbor_list:
                atom3 = atom_list[atom3_index]
                if atom3.elem_type == "X":
                    continue
                if atom3_index in atom1.third_nearest_neighbor_list:
                    first_third_third_triplets_set.add(Cluster(atom1, atom2, atom3))
        for atom2_index in atom1.second_nearest_neighbor_list:
            atom2 = atom_list[atom2_index]
            if atom2.elem_type == "X":
                continue
            for atom3_index in atom2.third_nearest_neighbor_list:
                atom3 = atom_list[atom3_index]
                if atom3.elem_type == "X":
                    continue
                if atom3_index in atom1.third_nearest_neighbor_list:
                    second_third_third_triplets_set.add(Cluster(atom1, atom2, atom3))
        for atom2_index in atom1.third_nearest_neighbor_list:
            atom2 = atom_list[atom2_index]
            if atom2.elem_type == "X":
                continue
            for atom3_index in atom2.third_nearest_neighbor_list:
                atom3 = atom_list[atom3_index]
                if atom3.elem_type == "X":
                    continue
                if atom3_index in atom1.third_nearest_neighbor_list:
                    third_third_third_triplets_set.add(Cluster(atom1, atom2, atom3))
    _get_average_parameters_mapping_from_cluster_vector_helper(list(first_first_first_triplets_set), cluster_mapping)
    _get_average_parameters_mapping_from_cluster_vector_helper(list(first_first_second_triplets_set), cluster_mapping)
    _get_average_parameters_mapping_from_cluster_vector_helper(list(first_first_third_triplets_set), cluster_mapping)
    _get_average_parameters_mapping_from_cluster_vector_helper(list(first_second_third_triplets_set), cluster_mapping)
    _get_average_parameters_mapping_from_cluster_vector_helper(list(first_third_third_triplets_set), cluster_mapping)
    _get_average_parameters_mapping_from_cluster_vector_helper(list(second_third_third_triplets_set), cluster_mapping)
    _get_average_parameters_mapping_from_cluster_vector_helper(list(third_third_third_triplets_set), cluster_mapping)

    # # quadruplets
    # first_kind_quadruplets_set: typing.Set[Cluster] = set()
    # second_kind_quadruplets_set: typing.Set[Cluster] = set()
    # third_kind_quadruplets_set: typing.Set[Cluster] = set()
    #
    # for atom1 in atom_list:
    #     if atom1.elem_type == "X":
    #         continue
    #     for atom2_index in atom1.first_nearest_neighbor_list:
    #         atom2 = atom_list[atom2_index]
    #         if atom2.elem_type == "X":
    #             continue
    #         for atom3_index in atom2.first_nearest_neighbor_list:
    #             if atom3_index not in atom1.first_nearest_neighbor_list:
    #                 continue
    #             atom3 = atom_list[atom3_index]
    #             if atom3.elem_type == "X":
    #                 continue
    #             for atom4_index in atom3.first_nearest_neighbor_list:
    #                 if atom4_index not in atom1.first_nearest_neighbor_list:
    #                     continue
    #                 atom4 = atom_list[atom4_index]
    #                 if atom4.elem_type == "X":
    #                     continue
    #                 if atom4_index in atom2.first_nearest_neighbor_list:
    #                     first_kind_quadruplets_set.add(Cluster(atom1, atom2, atom3, atom4))
    #                 if atom4_index in atom2.second_nearest_neighbor_list:
    #                     second_kind_quadruplets_set.add(Cluster(atom1, atom2, atom3, atom4))
    #                 if atom4_index in atom2.third_nearest_neighbor_list:
    #                     third_kind_quadruplets_set.add(Cluster(atom1, atom2, atom3, atom4))
    # _get_average_parameters_mapping_from_cluster_vector_helper(list(first_kind_quadruplets_set), cluster_mapping)
    # _get_average_parameters_mapping_from_cluster_vector_helper(list(second_kind_quadruplets_set), cluster_mapping)
    # _get_average_parameters_mapping_from_cluster_vector_helper(list(third_kind_quadruplets_set), cluster_mapping)

    return cluster_mapping


def get_one_hot_encoding_list_from_mapping(
        config: cfg.Config,
        type_set: typing.Set[str],
        cluster_mapping: typing.List[typing.List[typing.List[int]]]) -> \
        typing.List[float]:
    one_hot_encode_dict = generate_one_hot_encode_dict_for_type(type_set)
    atom_vector = _get_sorted_atom_vector(config)
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
    return encode_list


if __name__ == "__main__":
    config11 = cfg.read_config("../test/test_files/test.cfg")
    cl_mapping = get_average_cluster_parameters_mapping_periodic(config11)
    ce = get_one_hot_encoding_list_from_mapping(config11, {"Al", "Mg", "Zn"}, cl_mapping)
    print(len(ce))
