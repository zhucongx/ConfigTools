import typing
import numpy as np


class Atom(object):
    def __init__(self, atom_id: int, mass: float, elem_type: str, x: float,
                 y: float, z: float):
        self.__atom_id = atom_id
        self.__mass = mass
        self.__elem_type = elem_type
        self.__relative_position = np.array([x, y, z])
        self.__cartesian_position = np.array([x, y, z])
        self.__first_nearest_neighbor_list: typing.List[int] = list()
        self.__second_nearest_neighbor_list: typing.List[int] = list()
        self.__third_nearest_neighbor_list: typing.List[int] = list()

    @property
    def cartesian_position(self) -> np.ndarray:
        return self.__cartesian_position

    @property
    def relative_position(self) -> np.ndarray:
        return self.__relative_position

    @property
    def atom_id(self) -> int:
        return self.__atom_id

    @property
    def mass(self) -> float:
        return self.__mass

    @property
    def elem_type(self) -> str:
        return self.__elem_type

    @property
    def first_nearest_neighbor_list(self) -> typing.List[int]:
        return self.__first_nearest_neighbor_list

    @property
    def second_nearest_neighbor_list(self) -> typing.List[int]:
        return self.__second_nearest_neighbor_list

    @property
    def third_nearest_neighbor_list(self) -> typing.List[int]:
        return self.__third_nearest_neighbor_list

    @relative_position.setter
    def relative_position(self, position: np.ndarray):
        if position.shape != (3,):
            raise ValueError(f'input position size is not (3,) but {position.shape}')
        self.__relative_position = position

    @cartesian_position.setter
    def cartesian_position(self, position: np.ndarray):
        if position.shape != (3,):
            raise ValueError(f'input position size is not (3,) but {position.shape}')
        self.__cartesian_position = position

    @atom_id.setter
    def atom_id(self, atom_id: int):
        self.__atom_id = atom_id

    @elem_type.setter
    def elem_type(self, elem_type: str):
        self.__elem_type = elem_type

    def append_first_nearest_neighbor_list(self, index: int):
        self.__first_nearest_neighbor_list.append(index)

    def append_second_nearest_neighbor_list(self, index: int):
        self.__second_nearest_neighbor_list.append(index)

    def append_third_nearest_neighbor_list(self, index: int):
        self.__third_nearest_neighbor_list.append(index)

    def clean_neighbors_lists(self):
        self.__first_nearest_neighbor_list.clear()
        self.__second_nearest_neighbor_list.clear()
        self.__third_nearest_neighbor_list.clear()


def get_relative_distance_vector(atom1: Atom, atom2: Atom) -> np.ndarray:
    """

    Parameters
    ----------
    atom1
    atom2

    Returns
    -------
    np.ndarray
    """
    relative_distance_vector = atom1.relative_position - atom2.relative_position
    for i in range(3):
        if relative_distance_vector[i] >= 0.5:
            relative_distance_vector[i] -= 1
        elif relative_distance_vector[i] < -0.5:
            relative_distance_vector[i] += 1
    return relative_distance_vector
