from configtools.cfg.constants import *
from configtools.cfg.atom import Atom, get_average_fractional_position_atom, \
    get_fractional_distance_vector
from configtools.cfg.atomic_mass import get_atomic_mass
from collections import OrderedDict
import typing
import logging
import numpy as np
import copy


class Config(object):
    def __init__(self, basis: typing.Union[list, np.ndarray] = None,
                 atom_list: typing.List[Atom] = None):
        if basis is None:
            self._basis = np.array([0, 0, 0], [0, 0, 0], [0, 0, 0])
        else:
            if isinstance(basis, list):
                basis = np.array(basis, dtype=np.float64)
            if basis.shape != (3, 3):
                raise RuntimeError(f"input basis size is not (3, 3) but {basis.shape}")
            self._basis: np.ndarray = basis

        if atom_list is None:
            self._atom_list: typing.List[Atom] = list()
        else:
            self._atom_list: typing.List[Atom] = atom_list

    @property
    def number_atoms(self) -> int:
        return len(self._atom_list)

    @property
    def basis(self) -> np.ndarray:
        return self._basis

    @property
    def atom_list(self) -> typing.List[Atom]:
        return self._atom_list

    @basis.setter
    def basis(self, basis: typing.Union[list, np.ndarray]):
        if isinstance(basis, list):
            basis = np.array(basis, dtype=np.float64)
        if basis.shape != (3, 3):
            raise RuntimeError(f"input basis size is not (3, 3) but {basis.shape}")
        self._basis = basis

    def convert_fractional_to_cartesian(self) -> None:
        for i, atom in enumerate(self._atom_list):
            self._atom_list[i].cartesian_position = atom.fractional_position.dot(self._basis)

    def convert_cartesian_to_fractional(self) -> None:
        inverse_basis = np.linalg.inv(self._basis)
        for i, atom in enumerate(self._atom_list):
            self._atom_list[i].fractional_position = atom.cartesian_position.dot(inverse_basis)

    def perturb(self) -> None:
        inverse_basis = np.linalg.inv(self._basis)
        for i, atom in enumerate(self._atom_list):
            self._atom_list[i].fractional_position = (
                    atom.cartesian_position + np.random.normal(0, 0.1, 3)).dot(
                inverse_basis)

    def warp_at_periodic_boundaries(self) -> None:
        for i, atom in enumerate(self._atom_list):
            fractional_position = atom.fractional_position
            fractional_position -= np.floor(fractional_position)
            self._atom_list[i].fractional_position = fractional_position
        self.convert_fractional_to_cartesian()

    def clear_neighbors(self) -> None:
        for atom in self._atom_list:
            atom.clean_neighbors_lists()

    def get_element_list_map(self) -> typing.OrderedDict[str, typing.List[int]]:
        element_list_map: typing.Dict[str, typing.List[int]] = dict()
        for atom in self._atom_list:
            try:
                element_list_map[atom.elem_type].append(atom.atom_id)
            except KeyError:
                element_list_map[atom.elem_type] = [atom.atom_id]
        return OrderedDict(sorted(element_list_map.items(), key=lambda it: it[0]))

    def update_neighbors(self) -> None:
        for i in range(self.number_atoms):
            for j in range(i):
                fractional_distance_vector = get_fractional_distance_vector(self._atom_list[i],
                                                                            self._atom_list[j])
                absolute_distance_vector = fractional_distance_vector.dot(self._basis)

                if abs(absolute_distance_vector[0]) > SEVENTH_NEAREST_NEIGHBORS_CUTOFF:
                    continue
                if abs(absolute_distance_vector[1]) > SEVENTH_NEAREST_NEIGHBORS_CUTOFF:
                    continue
                if abs(absolute_distance_vector[2]) > SEVENTH_NEAREST_NEIGHBORS_CUTOFF:
                    continue
                absolute_distance_square = np.inner(absolute_distance_vector,
                                                    absolute_distance_vector)
                if absolute_distance_square <= SEVENTH_NEAREST_NEIGHBORS_CUTOFF ** 2:
                    if absolute_distance_square <= SIXTH_NEAREST_NEIGHBORS_CUTOFF ** 2:
                        if absolute_distance_square <= FIFTH_NEAREST_NEIGHBORS_CUTOFF ** 2:
                            if absolute_distance_square <= FOURTH_NEAREST_NEIGHBORS_CUTOFF ** 2:
                                if absolute_distance_square <= THIRD_NEAREST_NEIGHBORS_CUTOFF ** 2:
                                    if absolute_distance_square <= SECOND_NEAREST_NEIGHBORS_CUTOFF_U ** 2:
                                        if absolute_distance_square <= FIRST_NEAREST_NEIGHBORS_CUTOFF ** 2:
                                            self._atom_list[i].append_first_nearest_neighbor_list(j)
                                            self._atom_list[j].append_first_nearest_neighbor_list(i)
                                        elif absolute_distance_square <= SECOND_NEAREST_NEIGHBORS_CUTOFF_L ** 2:
                                            self._atom_list[i].append_second_nearest_neighbor_list(
                                                j)
                                            self._atom_list[j].append_second_nearest_neighbor_list(
                                                i)
                                    else:
                                        self._atom_list[i].append_third_nearest_neighbor_list(j)
                                        self._atom_list[j].append_third_nearest_neighbor_list(i)
                                else:
                                    self._atom_list[i].append_fourth_nearest_neighbor_list(j)
                                    self._atom_list[j].append_fourth_nearest_neighbor_list(i)
                            else:
                                self._atom_list[i].append_fifth_nearest_neighbor_list(j)
                                self._atom_list[j].append_fifth_nearest_neighbor_list(i)
                        else:
                            self._atom_list[i].append_sixth_nearest_neighbor_list(j)
                            self._atom_list[j].append_sixth_nearest_neighbor_list(i)
                    else:
                        self._atom_list[i].append_seventh_nearest_neighbor_list(j)
                        self._atom_list[j].append_seventh_nearest_neighbor_list(i)


def read_config(filename: str, update_neighbors: bool = True) -> Config:
    with open(filename, "r") as f:
        content_list = f.readlines()

    first_line_list = content_list[0].split()
    num_atoms = int(first_line_list[first_line_list.index("=") + 1])
    scale = float(content_list[1].split()[2])
    basis_xx = float(content_list[2].split()[2])
    basis_xy = float(content_list[3].split()[2])
    basis_xz = float(content_list[4].split()[2])
    basis_yx = float(content_list[5].split()[2])
    basis_yy = float(content_list[6].split()[2])
    basis_yz = float(content_list[7].split()[2])
    basis_zx = float(content_list[8].split()[2])
    basis_zy = float(content_list[9].split()[2])
    basis_zz = float(content_list[10].split()[2])
    basis = np.array([[basis_xx, basis_xy, basis_xz],
                      [basis_yx, basis_yy, basis_yz],
                      [basis_zx, basis_zy, basis_zz]], dtype=np.float64) * scale
    data_index = 10 + float(content_list[2].split('=')[-1])

    atom_list: typing.List[Atom] = list()
    id_count = 0
    neighbor_found = False
    while id_count < num_atoms:
        mass = float(content_list[data_index].split()[0])
        data_index += 1
        elem_type = content_list[data_index].split()[0]
        data_index += 1
        positions = content_list[data_index].split()
        data_index += 1

        atom = Atom(id_count, mass, elem_type, float(positions[0]), float(positions[1]),
                    float(positions[2]))
        try:
            if positions[3] == "#":
                base_index = 4
                for i in range(NUM_FIRST_NEAREST_NEIGHBORS):
                    atom.append_first_nearest_neighbor_list(int(positions[base_index + i]))
                base_index += NUM_FIRST_NEAREST_NEIGHBORS
                for i in range(NUM_SECOND_NEAREST_NEIGHBORS):
                    atom.append_second_nearest_neighbor_list(int(positions[base_index + i]))
                base_index += NUM_SECOND_NEAREST_NEIGHBORS
                for i in range(NUM_THIRD_NEAREST_NEIGHBORS):
                    atom.append_third_nearest_neighbor_list(int(positions[base_index + i]))
                # for i in range(NUM_FOURTH_NEAREST_NEIGHBORS):
                #     atom.append_fourth_nearest_neighbor_list(int(positions[base_index + i]))
                neighbor_found = True
        except IndexError:
            pass
        atom_list.append(atom)
        id_count += 1

    config = Config(basis, atom_list)
    config.warp_at_periodic_boundaries()
    logging.debug(f"Found neighbors {neighbor_found}")
    if (not neighbor_found) and update_neighbors:
        logging.debug(f"Finding neighbors")
        config.update_neighbors()

    return config


def write_config(config: Config, filename: str, neighbors_info: bool = True) -> None:
    content = "Number of particles = " + str(
        config.number_atoms) + "\nA = 1.0 Angstrom (basic length-scale)\n"
    content += f"H0(1,1) = {config.basis[0][0]} A\nH0(1,2) = {config.basis[0][1]} A\nH0(1,3) = {config.basis[0][2]} A\n"
    content += f"H0(2,1) = {config.basis[1][0]} A\nH0(2,2) = {config.basis[1][1]} A\nH0(2,3) = {config.basis[1][2]} A\n"
    content += f"H0(3,1) = {config.basis[2][0]} A\nH0(3,2) = {config.basis[2][1]} A\nH0(3,3) = {config.basis[2][2]} A\n"
    content += ".NO_VELOCITY.\nentry_count = 3\n"
    for atom in config.atom_list:
        content += str(atom.mass) + "\n" + atom.elem_type + "\n"
        content += np.array2string(atom.fractional_position,
                                   formatter={"float_kind": lambda x: "%.16f" % x})[1:-1]
        if neighbors_info:
            content += " # "
            content += "".join(
                str(index) + " " for index in
                atom.first_nearest_neighbor_list + atom.second_nearest_neighbor_list +
                atom.third_nearest_neighbor_list + atom.fourth_nearest_neighbor_list)
        content += "\n"

    with open(filename, "w") as f:
        f.write(content)


def read_poscar(filename: str, update_neighbors: bool = True) -> Config:
    with open(filename, "r") as f:
        content_list = f.readlines()

    # get scale factor ad
    scale = float(content_list[1])

    # bases
    basis = [vector.split() for vector in content_list[2:5]]
    basis = np.array(basis, dtype=np.float64) * scale

    elem_types_list = content_list[5].split()
    elem_numbers_list = content_list[6].split()
    elem_numbers_list = [int(i) for i in elem_numbers_list]

    if content_list[7][0] in "Ss":
        fractional_option = content_list[8][0] in "Dd"
        data_begin_index = 9
    else:
        fractional_option = content_list[7][0] in "Dd"
        data_begin_index = 8

    atom_list: typing.List[Atom] = list()
    id_count = 0
    for elem_type, elem_number in zip(elem_types_list, elem_numbers_list):
        mass = get_atomic_mass(elem_type)
        for num_count in range(elem_number):
            positions = content_list[data_begin_index + id_count].split()
            atom = Atom(id_count, mass, elem_type, float(positions[0]), float(positions[1]),
                        float(positions[2]))
            atom_list.append(atom)
            id_count += 1

    config = Config(basis, atom_list)
    if fractional_option:
        config.convert_fractional_to_cartesian()
    else:
        config.convert_cartesian_to_fractional()
    config.warp_at_periodic_boundaries()

    if update_neighbors:
        logging.debug(f"Finding neighbors")
        config.update_neighbors()
    return config


def write_poscar(config: Config, filename: str) -> None:
    content = "#comment\n1.0\n"
    for basis_row in config.basis:
        for base in basis_row:
            content += f"{base} "
        content += "\n"
    element_list_map = config.get_element_list_map()

    element_str = ""
    count_str = ""
    for element, element_list in element_list_map.items():
        if element == "X":
            continue
        element_str += element + " "
        count_str += str(len(element_list)) + " "
    content += element_str + "\n" + count_str + "\n"
    content += "Direct\n"
    for element, element_list in element_list_map.items():
        if element == "X":
            continue
        for index in element_list:
            content += np.array2string(config.atom_list[int(index)].fractional_position,
                                       formatter={"float_kind": lambda x: "%.16f" % x})[1:-1] + "\n"
    with open(filename, "w") as f:
        f.write(content)


def get_average_position_config(config1: Config, config2: Config) -> Config:
    if config1.number_atoms != config2.number_atoms:
        raise RuntimeError(
            f"first config has {config1.number_atoms} atoms but second has {config2.number_atoms}")
    if np.linalg.norm(config1.basis - config2.basis) > 1e-6:
        raise RuntimeError(f"base vectors do not match")
    config1.clear_neighbors()
    config2.clear_neighbors()
    atom_list: typing.List[Atom] = list()
    for atom1, atom2 in zip(config1.atom_list, config2.atom_list):
        atom_list.append(get_average_fractional_position_atom(atom1, atom2))
    res = Config(config1.basis, atom_list)
    res.convert_fractional_to_cartesian()
    return res


def get_pair_center(config: Config, jump_pair: typing.Tuple[int, int]) -> np.ndarray:
    first, second = jump_pair
    center_position = np.zeros(3)
    for kDim in range(3):
        first_fractional: float = config.atom_list[first].fractional_position[kDim]
        second_fractional: float = config.atom_list[second].fractional_position[kDim]
        distance: float = first_fractional - second_fractional
        period: int = int(distance / 0.5)
        # make sure distance is in the range (0, 0.5)
        while period != 0:
            first_fractional -= float(period)
            distance = first_fractional - second_fractional
            period = int(distance / 0.5)
        center_position[kDim] = 0.5 * (first_fractional + second_fractional)
    return center_position


def get_pair_rotation_matrix(config: Config, jump_pair: typing.Tuple[int, int]) -> np.ndarray:
    first, second = jump_pair
    first_atom: Atom = config.atom_list[first]

    pair_direction = get_fractional_distance_vector(first_atom, config.atom_list[second])
    logging.debug(f"{pair_direction}")
    pair_direction /= np.linalg.norm(pair_direction)
    vertical_vector = np.zeros(3)
    for index in first_atom.first_nearest_neighbor_list:
        jump_vector = get_fractional_distance_vector(first_atom, config.atom_list[index])
        # to exam if jump_vector and pair_direction perpendicular
        dot_prod = np.dot(jump_vector, pair_direction)
        if abs(dot_prod) < 1e-6:
            logging.debug(f"dot_prod value is {dot_prod}")
            vertical_vector = jump_vector
            vertical_vector /= np.linalg.norm(vertical_vector)
            break
    # The third row is normalized since it is a cross product of two normalized vectors.
    # We use transposed matrix here because transpose of an orthogonal matrix equals its inverse
    return np.array(
        (pair_direction, vertical_vector, np.cross(pair_direction, vertical_vector))).transpose()


def get_all_neighbors_set_of_jump_pair(config: Config, jump_pair: typing.Tuple[int, int]) -> \
        typing.Set[int]:
    near_neighbors_hashset: typing.Set[int] = set()
    for i in jump_pair:
        atom = config.atom_list[i]
        for j in atom.first_nearest_neighbor_list + atom.second_nearest_neighbor_list + \
                 atom.third_nearest_neighbor_list + atom.fourth_nearest_neighbor_list + \
                 atom.fifth_nearest_neighbor_list + atom.sixth_nearest_neighbor_list + \
                 atom.seventh_nearest_neighbor_list:
            near_neighbors_hashset.add(j)
    return near_neighbors_hashset


def get_neighbors_set_of_jump_pair(
        config: Config, jump_pair: typing.Tuple[int, int]) -> typing.Set[int]:
    near_neighbors_hashset: typing.Set[int] = set()
    for i in jump_pair:
        atom = config.atom_list[i]
        for j in atom.first_nearest_neighbor_list + atom.second_nearest_neighbor_list + atom.third_nearest_neighbor_list:
            near_neighbors_hashset.add(j)
    return near_neighbors_hashset


def get_neighbors_set_of_atom(
        config: Config, atom_id: int) -> typing.Set[int]:
    near_neighbors_hashset: typing.Set[int] = set()
    atom = config.atom_list[atom_id]
    for j in atom.first_nearest_neighbor_list + atom.second_nearest_neighbor_list + atom.third_nearest_neighbor_list:
        near_neighbors_hashset.add(j)
    return near_neighbors_hashset


def get_vacancy_index(config: Config) -> int:
    for atom in config.atom_list:
        if atom.elem_type == "X":
            return atom.atom_id
    raise NotImplementedError("No vacancy found")


def get_neighbors_set_of_vacancy(config: Config, vacancy_index: int) -> typing.Set[int]:
    near_neighbors_hashset: typing.Set[int] = set()
    near_neighbors_hashset.add(vacancy_index)

    for i in range(5):
        new_set = copy.deepcopy(near_neighbors_hashset)
        for index in new_set:
            atom = config.atom_list[index]
            for j in atom.first_nearest_neighbor_list:
                near_neighbors_hashset.add(j)

    return near_neighbors_hashset


def rotate_atom_vector(atom_list: typing.List[Atom], rotation_matrix: np.ndarray) -> None:
    move_distance_after_rotation = np.full((3,), 0.5) - np.full((3,), 0.5).dot(rotation_matrix)

    for i, atom in enumerate(atom_list):
        fractional_position = atom.fractional_position
        fractional_position = fractional_position.dot(rotation_matrix)
        fractional_position += move_distance_after_rotation
        fractional_position -= np.floor(fractional_position)

        atom_list[i].fractional_position = fractional_position


def get_config_system(config: Config) -> str:
    type_set = set()
    for atom in config.atom_list:
        if atom.elem_type == "X":
            continue
        type_set.add(atom.elem_type)
    type_list = sorted(list(type_set))
    return "-".join(type_list)


def find_jump_pair_from_cfg(config_start: Config, config_end: Config) -> typing.Tuple[int, int]:
    index_distance_list = list()
    for atom1, atom2 in zip(config_start.atom_list, config_end.atom_list):
        fractional_distance_vector = get_fractional_distance_vector(atom1, atom2)
        fractional_distance_square = np.inner(fractional_distance_vector,
                                              fractional_distance_vector)
        index_distance_list.append((atom1, fractional_distance_square))
    index_distance_list.sort(key=lambda sort_pair: sort_pair[1], reverse=True)
    if index_distance_list[0][0].elem_type == "X":
        return index_distance_list[0][0].atom_id, index_distance_list[1][0].atom_id
    else:
        return index_distance_list[1][0].atom_id, index_distance_list[0][0].atom_id


def find_jump_id_from_poscar(config_start: Config, config_end: Config) -> int:
    index_distance_list = list()
    for atom1, atom2 in zip(config_start.atom_list, config_end.atom_list):
        fractional_distance_vector = get_fractional_distance_vector(atom1, atom2)
        fractional_distance_square = np.inner(fractional_distance_vector,
                                              fractional_distance_vector)
        index_distance_list.append((atom1, fractional_distance_square))
    index_distance_list.sort(key=lambda sort_pair: sort_pair[1], reverse=True)
    return index_distance_list[0][0].atom_id


def get_distance_of_atom_between(config_start: Config, config_end: Config, atom_id: int) -> float:
    fractional_distance_vector = config_end.atom_list[atom_id].fractional_position - \
                                 config_start.atom_list[atom_id].fractional_position
    for i in range(3):
        while fractional_distance_vector[i] >= 0.5:
            fractional_distance_vector[i] -= 1
        while fractional_distance_vector[i] < -0.5:
            fractional_distance_vector[i] += 1
    absolute_distance_vector = fractional_distance_vector.dot(config_start.basis)
    absolute_distance_square = np.inner(absolute_distance_vector, absolute_distance_vector)
    return np.sqrt(absolute_distance_square)


def get_fractional_distance_vector_of_atom_between(config_start: Config, config_end: Config,
                                                   atom_id: int) -> np.ndarray:
    fractional_distance_vector = config_end.atom_list[atom_id].fractional_position - \
                                 config_start.atom_list[atom_id].fractional_position
    for i in range(3):
        while fractional_distance_vector[i] >= 0.5:
            fractional_distance_vector[i] -= 1
        while fractional_distance_vector[i] < -0.5:
            fractional_distance_vector[i] += 1
    return fractional_distance_vector


def atoms_jump(config: Config, jump_pair: typing.Tuple[int, int]):
    lhs, rhs = jump_pair
    temp = config.atom_list[lhs].fractional_position
    config.atom_list[lhs].fractional_position = config.atom_list[rhs].fractional_position
    config.atom_list[rhs].fractional_position = temp

    temp = config.atom_list[lhs].cartesian_position
    config.atom_list[lhs].cartesian_position = config.atom_list[rhs].cartesian_position
    config.atom_list[rhs].cartesian_position = temp

    temp = config.atom_list[lhs].first_nearest_neighbor_list
    config.atom_list[lhs].first_nearest_neighbor_list = config.atom_list[
        rhs].first_nearest_neighbor_list
    config.atom_list[rhs].first_nearest_neighbor_list = temp

    temp = config.atom_list[lhs].second_nearest_neighbor_list
    config.atom_list[lhs].second_nearest_neighbor_list = config.atom_list[
        rhs].second_nearest_neighbor_list
    config.atom_list[rhs].second_nearest_neighbor_list = temp

    temp = config.atom_list[lhs].third_nearest_neighbor_list
    config.atom_list[lhs].third_nearest_neighbor_list = config.atom_list[
        rhs].third_nearest_neighbor_list
    config.atom_list[rhs].third_nearest_neighbor_list = temp

    temp = config.atom_list[lhs].fourth_nearest_neighbor_list
    config.atom_list[lhs].fourth_nearest_neighbor_list = config.atom_list[
        rhs].fourth_nearest_neighbor_list
    config.atom_list[rhs].fourth_nearest_neighbor_list = temp

    temp = config.atom_list[lhs].fifth_nearest_neighbor_list
    config.atom_list[lhs].fifth_nearest_neighbor_list = config.atom_list[
        rhs].fifth_nearest_neighbor_list
    config.atom_list[rhs].fifth_nearest_neighbor_list = temp

    temp = config.atom_list[lhs].sixth_nearest_neighbor_list
    config.atom_list[lhs].sixth_nearest_neighbor_list = config.atom_list[
        rhs].sixth_nearest_neighbor_list
    config.atom_list[rhs].sixth_nearest_neighbor_list = temp

    temp = config.atom_list[lhs].seventh_nearest_neighbor_list
    config.atom_list[lhs].seventh_nearest_neighbor_list = config.atom_list[
        rhs].seventh_nearest_neighbor_list
    config.atom_list[rhs].seventh_nearest_neighbor_list = temp

    atom_id_set = get_all_neighbors_set_of_jump_pair(config, jump_pair)
    for i in atom_id_set:
        for index, j in enumerate(config.atom_list[i].first_nearest_neighbor_list):
            if j == lhs:
                config.atom_list[i].first_nearest_neighbor_list[index] = rhs
            if j == rhs:
                config.atom_list[i].first_nearest_neighbor_list[index] = lhs
        for index, j in enumerate(config.atom_list[i].second_nearest_neighbor_list):
            if j == lhs:
                config.atom_list[i].second_nearest_neighbor_list[index] = rhs
            if j == rhs:
                config.atom_list[i].second_nearest_neighbor_list[index] = lhs
        for index, j in enumerate(config.atom_list[i].third_nearest_neighbor_list):
            if j == lhs:
                config.atom_list[i].third_nearest_neighbor_list[index] = rhs
            if j == rhs:
                config.atom_list[i].third_nearest_neighbor_list[index] = lhs
        for index, j in enumerate(config.atom_list[i].fourth_nearest_neighbor_list):
            if j == lhs:
                config.atom_list[i].fourth_nearest_neighbor_list[index] = rhs
            if j == rhs:
                config.atom_list[i].fourth_nearest_neighbor_list[index] = lhs
        for index, j in enumerate(config.atom_list[i].fifth_nearest_neighbor_list):
            if j == lhs:
                config.atom_list[i].fifth_nearest_neighbor_list[index] = rhs
            if j == rhs:
                config.atom_list[i].fifth_nearest_neighbor_list[index] = lhs
        for index, j in enumerate(config.atom_list[i].sixth_nearest_neighbor_list):
            if j == lhs:
                config.atom_list[i].sixth_nearest_neighbor_list[index] = rhs
            if j == rhs:
                config.atom_list[i].sixth_nearest_neighbor_list[index] = lhs
        for index, j in enumerate(config.atom_list[i].seventh_nearest_neighbor_list):
            if j == lhs:
                config.atom_list[i].seventh_nearest_neighbor_list[index] = rhs
            if j == rhs:
                config.atom_list[i].seventh_nearest_neighbor_list[index] = lhs


def generate_fcc(elem_type: str, lattice_constant: float, factor: int) -> Config:
    basis = np.array([[1., 0., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.]], dtype=np.float64) * lattice_constant * factor
    atom_list: typing.List[Atom] = list()
    ct = 0
    for i in range(factor):
        for j in range(factor):
            for k in range(factor):
                atom_list.append(
                    Atom(ct, get_atomic_mass(elem_type), elem_type, i * lattice_constant,
                         j * lattice_constant, k * lattice_constant))
                ct += 1
                atom_list.append(
                    Atom(ct, get_atomic_mass(elem_type), elem_type, (i + 0.5) * lattice_constant,
                         (j + 0.5) * lattice_constant, k * lattice_constant))
                ct += 1
                atom_list.append(
                    Atom(ct, get_atomic_mass(elem_type), elem_type, i * lattice_constant,
                         (j + 0.5) * lattice_constant, (k + 0.5) * lattice_constant))
                ct += 1
                atom_list.append(
                    Atom(ct, get_atomic_mass(elem_type), elem_type, (i + 0.5) * lattice_constant,
                         j * lattice_constant, (k + 0.5) * lattice_constant))
                ct += 1

    config = Config(basis, atom_list)
    config.convert_cartesian_to_fractional()
    return config


def remove_fcc_random_displacement(config: Config, lattice_constant: float, factor: int) -> Config:
    reference_config = generate_fcc(config.atom_list[0].elem_type, lattice_constant, factor)
    for atom1 in config.atom_list:
        min_distance = 1e10
        min_index = -1
        for atom2 in reference_config.atom_list:
            fractional_distance_vector = atom1.fractional_position - atom2.fractional_position
            for i in range(3):
                while fractional_distance_vector[i] >= 0.5:
                    fractional_distance_vector[i] -= 1
                while fractional_distance_vector[i] < -0.5:
                    fractional_distance_vector[i] += 1
            absolute_distance_vector = fractional_distance_vector.dot(config.basis)
            absolute_distance_square = np.inner(absolute_distance_vector, absolute_distance_vector)
            if absolute_distance_square < min_distance:
                min_distance = absolute_distance_square
                min_index = atom2.atom_id
        atom1.fractional_position = reference_config.atom_list[min_index].fractional_position
        atom1.cartesian_position = reference_config.atom_list[min_index].cartesian_position
    return config

# if __name__ == "__main__":
#    config1 = read_config("../../test/test_files/forward.cfg")
#    atoms_jump(config1, (18, 23))
#    config2 = read_config("../../test/test_files/backward.cfg")
#     print(find_jump_pair(config1, config2))

# config = read_config("../test/test_files/test.cfg")
# vacancy_id = get_vacancy_index(config)
# print(vacancy_id)
# move_distance = np.full((3,), 0.5) - config.atom_list[vacancy_id].fractional_position
#
# atom_set = get_neighbors_set_of_vacancy(config, vacancy_id)
# atom_list = config.atom_list
# new_atom_list = list()
# for idx in atom_set:
#     atom = atom_list[idx]
#     fractional_position = atom.fractional_position
#     fractional_position += move_distance
#     fractional_position -= np.floor(fractional_position)
#     atom.fractional_position = fractional_position
#     cartesian_position = (fractional_position - np.full((3,), 0.5)).dot(config.basis)
#     new_atom_list.append(atom_list[idx])
# new_config = Config(config.basis, new_atom_list)
# write_config(new_config, "test.cfg")
