from constants import *
from atom import *
from atomic_mass import get_atomic_mass
import numpy as np
import typing
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:: %(message)s')


class Config(object):
    def __init__(self, basis: typing.Union[list, np.ndarray] = None, atom_list: typing.List[Atom] = None):
        if basis is None:
            self.__basis = np.array([0, 0, 0], [0, 0, 0], [0, 0, 0])
        else:
            if isinstance(basis, list):
                basis = np.array(basis, dtype=np.float64)
            if basis.shape != (3, 3):
                raise ValueError(f"input basis size is not (3, 3) but {basis.shape}")
            self.__basis: np.ndarray = basis

        if atom_list is None:
            self.__atom_list: typing.List[Atom] = list()
        else:
            self.__atom_list: typing.List[Atom] = atom_list

        self.__logger = logging.getLogger("config_tools.Config")

    @property
    def number_atoms(self) -> int:
        return len(self.__atom_list)

    @property
    def basis(self) -> np.ndarray:
        return self.__basis

    @property
    def atom_list(self) -> typing.List[Atom]:
        return self.__atom_list

    @basis.setter
    def basis(self, basis: typing.Union[list, np.ndarray]):
        if isinstance(basis, list):
            basis = np.array(basis, dtype=np.float64)
        self.__basis = basis

    def convert_relative_to_cartesian(self):
        for i, atom in enumerate(self.__atom_list):
            atom.cartesian_position = atom.relative_position.dot(self.__basis)
            self.__atom_list[i] = atom

    def convert_cartesian_to_relative(self):
        inverse_basis = np.linalg.inv(self.__basis)
        for i, atom in enumerate(self.__atom_list):
            atom.relative_position = atom.cartesian_position.dot(inverse_basis)
            self.__atom_list[i] = atom

    def clear_neighbors(self):
        for atom in self.__atom_list:
            atom.clean_neighbors_lists()

    def update_neighbors(self, first_r_cutoff: float = FIRST_NEAREST_NEIGHBORS_CUTOFF,
                         second_r_cutoff: float = SECOND_NEAREST_NEIGHBORS_CUTOFF,
                         third_r_cutoff: float = THIRD_NEAREST_NEIGHBORS_CUTOFF):
        first_r_cutoff_square = first_r_cutoff * first_r_cutoff
        second_r_cutoff_square = second_r_cutoff * second_r_cutoff
        third_r_cutoff_square = third_r_cutoff * third_r_cutoff

        for i in range(self.number_atoms):
            for j in range(i):
                relative_distance_vector = get_relative_distance_vector(self.__atom_list[i],
                                                                        self.__atom_list[j])
                absolute_distance_vector = relative_distance_vector.dot(self.__basis)

                if abs(absolute_distance_vector[0]) > third_r_cutoff:
                    continue
                if abs(absolute_distance_vector[1]) > third_r_cutoff:
                    continue
                if abs(absolute_distance_vector[2]) > third_r_cutoff:
                    continue
                absolute_distance_square = np.inner(absolute_distance_vector, absolute_distance_vector)
                if absolute_distance_square <= third_r_cutoff_square:
                    if absolute_distance_square <= second_r_cutoff_square:
                        if absolute_distance_square <= first_r_cutoff_square:
                            self.__atom_list[i].append_first_nearest_neighbor_list(j)
                            self.__atom_list[j].append_first_nearest_neighbor_list(i)
                        else:
                            self.__atom_list[i].append_second_nearest_neighbor_list(j)
                            self.__atom_list[j].append_second_nearest_neighbor_list(i)
                    else:
                        self.__atom_list[i].append_third_nearest_neighbor_list(j)
                        self.__atom_list[j].append_third_nearest_neighbor_list(i)


def read_config(filename: str, update_neighbors: bool = True) -> Config:
    with open(filename, 'r') as f:
        content_list = f.readlines()

    first_line_list = content_list[0].split()
    num_atoms = int(first_line_list[first_line_list.index('=') + 1])
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
    data_index = 13

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

        atom = Atom(id_count, mass, elem_type, float(positions[0]), float(positions[1]), float(positions[2]))
        try:
            if positions[3] == '#':
                base_index = 4
                for i in range(NUM_FIRST_NEAREST_NEIGHBORS):
                    atom.append_first_nearest_neighbor_list(int(positions[base_index + i]))
                base_index += NUM_FIRST_NEAREST_NEIGHBORS
                for i in range(NUM_SECOND_NEAREST_NEIGHBORS):
                    atom.append_second_nearest_neighbor_list(int(positions[base_index + i]))
                base_index += NUM_SECOND_NEAREST_NEIGHBORS
                for i in range(NUM_THIRD_NEAREST_NEIGHBORS):
                    atom.append_third_nearest_neighbor_list(int(positions[base_index + i]))
                neighbor_found = True
        except IndexError:
            pass
        atom_list.append(atom)
        id_count += 1

    config = Config(basis, atom_list)
    config.convert_relative_to_cartesian()
    logging.info(f"Found neighbors {neighbor_found}")
    if (not neighbor_found) and update_neighbors:
        config.update_neighbors()

    return config


def read_poscar(filename: str, update_neighbors: bool = True) -> Config:
    with open(filename, 'r') as f:
        content_list = f.readlines()

    # get scale factor ad
    scale = float(content_list[1])

    # bases
    basis = [vector.split() for vector in content_list[2:5]]
    basis = np.array(basis, dtype=np.float64) * scale

    elem_types_list = content_list[5].split()
    elem_numbers_list = content_list[6].split()
    elem_numbers_list = [int(i) for i in elem_numbers_list]

    if content_list[7][0] in 'Ss':
        relative_option = content_list[8][0] in 'Dd'
        data_begin_index = 9
    else:
        relative_option = content_list[7][0] in 'Dd'
        data_begin_index = 8

    atom_list: typing.List[Atom] = list()
    id_count = 0
    for elem_type, elem_number in zip(elem_types_list, elem_numbers_list):
        mass = get_atomic_mass(elem_type)
        for num_count in range(elem_number):
            positions = content_list[data_begin_index + id_count].split()
            atom = Atom(id_count, mass, elem_type, float(positions[0]), float(positions[1]), float(positions[2]))
            atom_list.append(atom)
            id_count += 1

    config = Config(basis, atom_list)
    if relative_option:
        config.convert_relative_to_cartesian()
    else:
        config.convert_cartesian_to_relative()

    if update_neighbors:
        config.update_neighbors()
    return config
