from .atom import *
import os
import numpy as np


class Config(object):
    def __init__(self, basis: np.ndarray = None, atom_list: typing.List[Atom] = None):
        if basis is None:
            self.__basis = np.array([0, 0, 0], [0, 0, 0], [0, 0, 0])
        else:
            if basis.shape != (3, 3):
                raise ValueError(f"input basis size is not (3, 3) but {basis.shape}")
            self.__basis: np.ndarray = basis

        if atom_list is None:
            self.__atom_list: typing.List[Atom] = typing.List[Atom]()
        else:
            self.__atom_list: typing.List[Atom] = atom_list

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
    def basis(self, basis: np.ndarray):
        self.__basis = basis

    def convert_relative_to_cartesian(self):
        for atom in self.__atom_list:
            atom.cartesian_position = atom.relative_position.dot(self.__basis)

    def convert_cartesian_to_relative(self):
        inverse_basis = np.linalg.inv(self.__basis)
        for atom in self.__atom_list:
            atom.relative_position = atom.cartesian_position.dot(inverse_basis)

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
                absolute_distance_vector = get_relative_distance_vector(self.__atom_list[i],
                                                                        self.__atom_list[j]).dot(self.__basis)
                if absolute_distance_vector[0] > third_r_cutoff_square:
                    continue
                if absolute_distance_vector[1] > third_r_cutoff_square:
                    continue
                if absolute_distance_vector[2] > third_r_cutoff_square:
                    continue
                absolute_distance_square = np.inner(absolute_distance_vector, absolute_distance_vector)
                if absolute_distance_square <= third_r_cutoff_square:
                    if absolute_distance_square <= second_r_cutoff_square:
                        if absolute_distance_square <= first_r_cutoff_square:
                            self.__atom_list[i].append_first_nearest_neighbor_list(self.__atom_list[j].atom_id)
                            self.__atom_list[j].append_first_nearest_neighbor_list(self.__atom_list[i].atom_id)
                        else:
                            self.__atom_list[i].append_second_nearest_neighbor_list(self.__atom_list[j].atom_id)
                            self.__atom_list[j].append_second_nearest_neighbor_list(self.__atom_list[i].atom_id)
                    else:
                        self.__atom_list[i].append_third_nearest_neighbor_list(self.__atom_list[j].atom_id)
                        self.__atom_list[j].append_third_nearest_neighbor_list(self.__atom_list[i].atom_id)


def read_config(filename: str) -> Config:
    with open(filename, 'r') as f:
        pass


def read_poscar(filename: str) -> Config:
    with open(filename, 'r') as f:
        content_list = f.readlines()

    # get scale factor ad
    scale = float(content_list[1])

    # bases
    bases = [basis.split() for basis in content_list[2:5]]

    # Atom info
    atom_types_list = content_list[5].split()
    # Atom number (str).
    atom_numbers_list = content_list[6].split()
    if content_list[7][0] in 'Ss':
        data_begin = 9
    else:
        data_begin = 8

    # get total number before load data
    atom_numbers = [int(i) for i in atom_numbers]
    natom = sum(atom_numbers)

    # data
    data, tf = [], []  # data and T or F info
    tf_dict = {}  # {tf: atom number}
    for line_str in content_list[data_begin: data_begin + natom]:
        line_list = str2list(line_str)
        data.append(line_list[:3])
        if len(line_list) > 3:
            tf_list = line_list[3:]
            tf.append(tf_list)
            # gather tf info to tf_dict
            tf_str = ','.join(tf_list)
            if tf_str not in tf_dict:
                tf_dict[tf_str] = 1
            else:
                tf_dict[tf_str] += 1
        else:
            tf.append(['T', 'T', 'T'])
            # gather tf info to tf_dict
            if 'T,T,T' not in tf_dict:
                tf_dict['T,T,T'] = 1
            else:
                tf_dict['T,T,T'] += 1

    # Data type convertion
    bases = np.float64(np.array(bases))  # to float
    data = np.float64(np.array(data))
    tf = np.array(tf)

    # set class attrs
    self.bases_const = bases_const
    self.bases = bases
    self.atom_types = atom_types
    self.atom_numbers = atom_numbers
    self.natom = natom
    self.data = data
    self.tf = tf
    self.totline = data_begin + natom  # total number of line
