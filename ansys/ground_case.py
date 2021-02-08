from cfg.config import *
from ansys.cluster import Cluster, generate_one_hot_encode_dict_for_type
from collections import OrderedDict
import bisect
import copy
import math

K_EPSILON = 1e-8


def _atom_sort_compare(lhs: Atom, rhs: Atom) -> bool:
    relative_position_lhs = lhs.relative_position - np.full((3,), 0.5)
    relative_position_rhs = rhs.relative_position - np.full((3,), 0.5)
    distance_square_lhs = np.inner(relative_position_lhs, relative_position_lhs)
    distance_square_rhs = np.inner(relative_position_rhs, relative_position_rhs)

    diff = distance_square_lhs - distance_square_rhs
    if diff < - K_EPSILON:
        return True
    if diff > K_EPSILON:
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
    return relative_position_lhs[2] < relative_position_rhs[2] - K_EPSILON


def _is_atom_smaller_symmetrically(lhs: Atom, rhs: Atom) -> bool:
    relative_position_lhs = lhs.relative_position - np.full((3,), 0.5)
    relative_position_rhs = rhs.relative_position - np.full((3,), 0.5)

    distance_square_lhs = np.inner(relative_position_lhs, relative_position_lhs)
    distance_square_rhs = np.inner(relative_position_rhs, relative_position_rhs)

    return distance_square_lhs < distance_square_rhs - K_EPSILON


def _is_cluster_smaller_symmetrically(lhs: Cluster, rhs: Cluster) -> bool:
    for atom1, atom2 in zip(lhs.atom_list, rhs.atom_list):
        if _is_atom_smaller_symmetrically(atom1, atom2):
            return True
        if _is_atom_smaller_symmetrically(atom2, atom1):
            return False
    # if it reaches here, it means that the clusters are same symmetrically.Returns false.
    return False


def _sort_helper(atom_list: typing.List[Atom], reference_config: Config) -> typing.List[Atom]:
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

    Atom.__lt__ = lambda self, other: _atom_sort_compare(self, other)
    atom_list.sort()

    for i, atom in enumerate(atom_list):
        atom_list[i].atom_id = i
    out_config = Config(reference_config.basis, atom_list)
    out_config.update_neighbors_nonperiodic()

    return out_config.atom_list


def get_symmetrically_sorted_atom_vectors(config: Config, jump_pair: typing.Tuple[int, int]) -> \
        typing.Tuple[typing.List[Atom], typing.List[Atom]]:
    # atom_id_set = get_more_neighbors_set_of_jump_pair(config, jump_pair)
    atom_id_set = set(range(256))
    move_distance_start = np.full((3,), 0.5) - config.atom_list[jump_pair[0]].relative_position
    move_distance_end = np.full((3,), 0.5) - config.atom_list[jump_pair[1]].relative_position

    atom_list_start: typing.List[Atom] = list()
    atom_list_end: typing.List[Atom] = list()

    for atom_id in atom_id_set:
        if atom_id == jump_pair[0]:
            continue
        atom = copy.deepcopy(config.atom_list[atom_id])
        atom.clean_neighbors_lists()
        relative_position = atom.relative_position
        relative_position += move_distance_start
        relative_position -= np.floor(relative_position)
        atom.relative_position = relative_position

        atom_list_start.append(atom)

    for atom_id in atom_id_set:
        if atom_id == jump_pair[0]:
            continue
        atom = copy.deepcopy(config.atom_list[atom_id])
        atom.clean_neighbors_lists()
        relative_position = atom.relative_position
        relative_position += move_distance_end
        relative_position -= np.floor(relative_position)
        atom.relative_position = relative_position
        if atom.atom_id == jump_pair[1]:
            atom.relative_position = np.full((3,), 0.5) - move_distance_start + move_distance_end
        atom_list_end.append(atom)

    return _sort_helper(atom_list_start, config), _sort_helper(atom_list_end, config)


# def get_symmetrically_sorted_atom_vector(config: Config, vacancy_id: int) -> typing.List[Atom]:
#     atom_id_set = get_neighbors_set_of_vacancy(config, vacancy_id)
#     move_distance = np.full((3,), 0.5) - config.atom_list[vacancy_id].relative_position
#     atom_list: typing.List[Atom] = list()
#
#     for atom_id in atom_id_set:
#         if atom_id == vacancy_id:
#             continue
#         atom = copy.deepcopy(config.atom_list[atom_id])
#         atom.clean_neighbors_lists()
#         relative_position = atom.relative_position
#         relative_position += move_distance
#         relative_position -= np.floor(relative_position)
#         atom.relative_position = relative_position
#
#         atom_list.append(atom)
#         # cartesian_position = (relative_position - np.full((3,), 0.5)).dot(config.basis)
#         # if np.sqrt(np.inner(cartesian_position, cartesian_position)) < 8.1:
#         #     atom_list.append(atom)
#
#     logging.debug(f'Init: {[atom.atom_id for atom in atom_list]}')
#     Atom.__lt__ = lambda self, other: _atom_sort_compare(self, other)
#     atom_list.sort()
#     logging.debug(f'Finial: {[atom.atom_id for atom in atom_list]}')
#
#     for i, atom in enumerate(atom_list):
#         atom_list[i].atom_id = i
#     out_config = Config(config.basis, atom_list)
#     out_config.update_neighbors()
#     return out_config.atom_list


def _get_average_parameters_mapping_from_cluster_vector_helper(
        cluster_list: typing.List[Cluster],
        cluster_mapping: typing.List[typing.List[typing.List[int]]]):
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


def get_average_cluster_parameters_mapping(config: Config) -> typing.List[typing.List[typing.List[int]]]:
    vacancy_index = get_vacancy_index(config)
    neighbor_index = config.atom_list[vacancy_index].first_nearest_neighbor_list[0]
    atom_vector = get_symmetrically_sorted_atom_vectors(config, (vacancy_index, neighbor_index))[0]
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

    # triplets
    fff_triplets_set: typing.Set[Cluster] = set()
    ffs_triplets_set: typing.Set[Cluster] = set()
    fft_triplets_set: typing.Set[Cluster] = set()

    for atom1 in atom_vector:
        for atom2_index in atom1.first_nearest_neighbor_list:
            atom2 = atom_vector[atom2_index]
            for atom3_index in atom2.first_nearest_neighbor_list:
                if atom3_index in atom1.first_nearest_neighbor_list:
                    fff_triplets_set.add(Cluster(atom1, atom2, atom_vector[atom3_index]))
                elif atom3_index in atom1.second_nearest_neighbor_list:
                    ffs_triplets_set.add(Cluster(atom1, atom2, atom_vector[atom3_index]))
                elif atom3_index in atom1.third_nearest_neighbor_list:
                    fft_triplets_set.add(Cluster(atom1, atom2, atom_vector[atom3_index]))
    _get_average_parameters_mapping_from_cluster_vector_helper(list(fff_triplets_set), cluster_mapping)
    _get_average_parameters_mapping_from_cluster_vector_helper(list(ffs_triplets_set), cluster_mapping)
    _get_average_parameters_mapping_from_cluster_vector_helper(list(fft_triplets_set), cluster_mapping)

    return cluster_mapping


def _element_wise_add_second_to_first(first_list: typing.List[float], second_list: typing.List[float]):
    if len(first_list) != len(second_list):
        raise RuntimeError("Size mismatch")
    for i in range(len(first_list)):
        first_list[i] += second_list[i]


def _element_wise_divide_float_from_list(float_list: typing.List[float], divisor: float):
    for i in range(len(float_list)):
        float_list[i] /= divisor


def get_one_hot_encoding_list_map(
        config: Config, jump_pair: typing.Tuple[int, int],
        type_set: typing.Set[str],
        cluster_mapping: typing.List[typing.List[typing.List[int]]]) -> \
        typing.Tuple[typing.List[float], typing.List[float]]:
    one_hot_encode_dict = generate_one_hot_encode_dict_for_type(type_set)

    result: typing.List[typing.List[float]] = list()
    atom_vectors = get_symmetrically_sorted_atom_vectors(config, jump_pair)
    for atom_vector in atom_vectors:
        encode_list: typing.List[float] = list()
        for cluster_vector in cluster_mapping:
            sum_of_list = [0.0] * (len(type_set) ** len(cluster_vector[0]))
            for cluster in cluster_vector:
                cluster_type_key = ''
                for atom_index in cluster:
                    cluster_type_key += atom_vector[atom_index].elem_type
                _element_wise_add_second_to_first(sum_of_list, one_hot_encode_dict[cluster_type_key])
            _element_wise_divide_float_from_list(sum_of_list, float(len(cluster_vector)))
            encode_list = encode_list + sum_of_list
        result.append(encode_list)
    return tuple(result)

# if __name__ == '__main__':
#     config = read_config("../test/test_files/test.cfg")
#     vacancy_id = get_vacancy_index(config)
#
#     atom_set = get_symmetrically_sorted_index(config, vacancy_id)
#     for a in atom_set:
#         print(a.index, a.distance)
