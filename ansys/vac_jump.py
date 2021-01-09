from cfg.config import *
from ansys.cluster import Cluster
from collections import OrderedDict
import bisect
import copy

# import logging


K_EPSILON = 1e-8


def get_vacancy_index(config: Config) -> int:
    for atom in config.atom_list:
        if atom.elem_type == "X":
            return atom.atom_id
    raise NotImplementedError('No vacancy found')


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


def _atom_sort_compare(lhs: Atom, rhs: Atom) -> bool:
    relative_position_lhs = lhs.relative_position
    relative_position_rhs = rhs.relative_position
    diff_x = relative_position_lhs[0] - relative_position_rhs[0]
    if diff_x < - K_EPSILON:
        return True
    if diff_x > K_EPSILON:
        return False
    diff_y_sym = abs(relative_position_lhs[1] - 0.5) - abs(relative_position_rhs[1] - 0.5)
    if diff_y_sym < - K_EPSILON:
        return True
    if diff_y_sym > K_EPSILON:
        return False
    diff_z_sym = abs(relative_position_lhs[2] - 0.5) - abs(relative_position_rhs[2] - 0.5)
    if diff_z_sym < - K_EPSILON:
        return True
    if diff_z_sym > K_EPSILON:
        return False
    diff_y = relative_position_lhs[1] - relative_position_rhs[1]
    if diff_y < - K_EPSILON:
        return True
    if diff_y > K_EPSILON:
        return False
    return relative_position_lhs[2] < relative_position_rhs[2] - K_EPSILON


def _is_atom_smaller_symmetrically(lhs: Atom, rhs: Atom) -> bool:
    relative_position_lhs = lhs.relative_position
    relative_position_rhs = rhs.relative_position
    diff_x = relative_position_lhs[0] - relative_position_rhs[0]
    if diff_x < - K_EPSILON:
        return True
    if diff_x > K_EPSILON:
        return False
    diff_y = abs(relative_position_lhs[1] - 0.5) - abs(relative_position_rhs[1] - 0.5)
    if diff_y < - K_EPSILON:
        return True
    if diff_y > K_EPSILON:
        return False

    return abs(relative_position_lhs[2] - 0.5) < abs(relative_position_rhs[2] - 0.5) - K_EPSILON


def _is_cluster_smaller_symmetrically(lhs: Cluster, rhs: Cluster) -> bool:
    for atom1, atom2 in zip(lhs.atom_list, rhs.atom_list):
        if _is_atom_smaller_symmetrically(atom1, atom2):
            return True
        if _is_atom_smaller_symmetrically(atom2, atom1):
            return False
    # if it reaches here, it means that the clusters are same symmetrically.Returns false.
    return False


def _rotate_atom_vector_and_sort_helper(atom_list: typing.List[Atom], reference_config: Config,
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
    rotate_atom_vector(atom_list, get_pair_rotation_matrix(reference_config, jump_pair))
    logging.debug(f'Init: {[atom.atom_id for atom in atom_list]}')

    Atom.__lt__ = lambda self, other: _atom_sort_compare(self, other)
    atom_list.sort()
    logging.debug(f'Finial: {[atom.atom_id for atom in atom_list]}')

    for i, atom in enumerate(atom_list):
        atom_list[i].atom_id = i
    config = Config(reference_config.basis, atom_list)
    config.update_neighbors()
    return config.atom_list


def get_symmetrically_sorted_atom_vectors(config: Config, jump_pair: typing.Tuple[int, int]) -> \
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
    atom_id_set = get_first_and_second_third_neighbors_set_of_jump_pair(config, jump_pair)
    move_distance = np.full((3,), 0.5) - get_pair_center(config, jump_pair)
    logging.debug(f'move_distance {move_distance}')
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

    # first nearest pairs
    first_pair_set: typing.Set[Cluster] = set()
    for atom1 in atom_vector:
        for atom2_index in atom1.first_nearest_neighbor_list:
            first_pair_set.add(Cluster(atom1, atom_vector[atom2_index]))

    _get_average_parameters_mapping_from_cluster_vector_helper(list(first_pair_set), cluster_mapping)

    # second nearest pairs
    second_pair_set: typing.Set[Cluster] = set()
    for atom1 in atom_vector:
        for atom2_index in atom1.second_nearest_neighbor_list:
            second_pair_set.add(Cluster(atom1, atom_vector[atom2_index]))

    _get_average_parameters_mapping_from_cluster_vector_helper(list(second_pair_set), cluster_mapping)

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
    return encode_dict


def get_average_cluster_parameters_forward_and_backward_from_map(
        config: Config, jump_pair: typing.Tuple[int, int],
        type_category_map: typing.Dict[str, float],
        cluster_mapping: typing.List[typing.List[typing.List[int]]]) -> \
        typing.Tuple[typing.List[float], typing.List[float]]:
    result: typing.List[typing.List[float]] = list()
    atom_vectors = get_symmetrically_sorted_atom_vectors(config, jump_pair)
    for atom_vector in atom_vectors:
        encode_list: typing.List[float] = list()
        encode_list.append(1.0)
        for cluster_vector in cluster_mapping:
            sum_of_functional = 0.0
            for cluster in cluster_vector:
                cumulative_product = 1.0
                for atom_index in cluster:
                    cumulative_product *= type_category_map[atom_vector[atom_index].elem_type]
                sum_of_functional += cumulative_product
            encode_list.append(sum_of_functional / len(cluster_vector))
        result.append(encode_list)
    return tuple(result)


def _element_wise_add_second_to_first(first_list: typing.List[float], second_list: typing.List[float]):
    if len(first_list) != len(second_list):
        raise RuntimeError("Size mismatch")
    for i in range(len(first_list)):
        first_list[i] += second_list[i]


def _element_wise_divide_float_from_list(float_list: typing.List[float], divisor: float):
    for i in range(len(float_list)):
        float_list[i] /= divisor


def get_one_hot_encoding_list_forward_and_backward_from_map(
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


if __name__ == '__main__':
    config = read_config("../test/test_files/test.cfg")
    cl_mapping = get_average_cluster_parameters_mapping(config)
    forward, backward = get_one_hot_encoding_list_forward_and_backward_from_map(
        config, (18, 23), {'Al', 'Mg', 'Zn'}, cl_mapping)
    print(len(forward))
