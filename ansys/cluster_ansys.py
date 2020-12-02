from cfg.config import *
from cfg.cluster import Cluster


def is_atom_smaller_symmetrically(lhs: Atom, rhs: Atom) -> bool:
    relative_position_lhs = lhs.relative_position
    relative_position_rhs = rhs.relative_position
    diff_x = relative_position_lhs[0] - relative_position_rhs[0]
    if diff_x < - 1e-10:
        return True
    if diff_x > 1e-10:
        return False
    diff_y = abs(relative_position_lhs[1] - 0.5) - abs(relative_position_rhs[1] - 0.5)
    if diff_y < - 1e-10:
        return True
    if diff_y > 1e-10:
        return False
    return abs(relative_position_lhs[2] - 0.5) < abs(relative_position_rhs[2] - 0.5) - 1e-10


def is_cluster_smaller_symmetrically(lhs: Cluster, rhs: Cluster) -> bool:
    for atom1, atom2 in zip(lhs.atom_list, rhs.atom_list):
        if is_atom_smaller_symmetrically(atom1, atom2):
            return True
        if is_atom_smaller_symmetrically(atom2, atom1):
            return False
    # if it reaches here, it means that the clusters are same symmetrically.Returns false.
    return False


def get_symmetrically_sorted_atom_vectors(config: Config, jump_pair: typing.Tuple[int, int]) -> \
        typing.Tuple[typing.List[Atom], typing.List[Atom]]:
    atom_id_set = get_first_and_second_third_neighbors_set_of_jump_pair(config, jump_pair)
    move_distance = np.full((3,), 0.5) - get_pair_center(config, jump_pair)
    atom_list_forward: typing.List[Atom] = list()
    vacancy_relative_position = np.zeros(3)
    vacancy_cartesian_position = np.zeros(3)
    for atom_id in atom_id_set:
        atom = config.atom_list[atom_id]
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
    atom_list_backward = atom_list_forward.copy()
    for i, atom in enumerate(atom_list_backward):
        if atom.atom_id == jump_pair[1]:
            atom_list_backward[i].relative_position = vacancy_relative_position
            atom_list_backward[i].cartesian_position = vacancy_cartesian_position


