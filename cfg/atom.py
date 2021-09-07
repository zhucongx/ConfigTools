import typing
import numpy as np


class Atom(object):
    def __init__(self, atom_id: int, mass: float, elem_type: str, x: float,
                 y: float, z: float):
        """
        A class to represent one atom.
        "copy.copy" of this class will make a copy of lists but np.darray type positions can still be changed.
        use "copy.deepcopy" instead
        Parameters
        ----------
        atom_id
        mass
        elem_type
        x
        y
        z
        """
        self._atom_id = atom_id
        self._mass = mass
        self._elem_type = elem_type
        self._relative_position = np.array([x, y, z])
        self._cartesian_position = np.array([x, y, z])
        self._first_nearest_neighbor_list: typing.List[int] = list()
        self._second_nearest_neighbor_list: typing.List[int] = list()
        self._third_nearest_neighbor_list: typing.List[int] = list()
        self._fourth_nearest_neighbor_list: typing.List[int] = list()
        self._fifth_nearest_neighbor_list: typing.List[int] = list()
        self._sixth_nearest_neighbor_list: typing.List[int] = list()
        self._seventh_nearest_neighbor_list: typing.List[int] = list()

    @property
    def cartesian_position(self) -> np.ndarray:
        return self._cartesian_position

    @property
    def relative_position(self) -> np.ndarray:
        return self._relative_position

    @property
    def atom_id(self) -> int:
        return self._atom_id

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def elem_type(self) -> str:
        return self._elem_type

    @property
    def first_nearest_neighbor_list(self) -> typing.List[int]:
        return self._first_nearest_neighbor_list

    @property
    def second_nearest_neighbor_list(self) -> typing.List[int]:
        return self._second_nearest_neighbor_list

    @property
    def third_nearest_neighbor_list(self) -> typing.List[int]:
        return self._third_nearest_neighbor_list

    @property
    def fourth_nearest_neighbor_list(self) -> typing.List[int]:
        return self._fourth_nearest_neighbor_list

    @property
    def fifth_nearest_neighbor_list(self) -> typing.List[int]:
        return self._fifth_nearest_neighbor_list

    @property
    def sixth_nearest_neighbor_list(self) -> typing.List[int]:
        return self._sixth_nearest_neighbor_list

    @property
    def seventh_nearest_neighbor_list(self) -> typing.List[int]:
        return self._seventh_nearest_neighbor_list

    @relative_position.setter
    def relative_position(self, position: np.ndarray):
        if position.shape != (3,):
            raise RuntimeError(f'input position size is not (3,) but {position.shape}')
        self._relative_position = position

    @cartesian_position.setter
    def cartesian_position(self, position: np.ndarray):
        if position.shape != (3,):
            raise RuntimeError(f'input position size is not (3,) but {position.shape}')
        self._cartesian_position = position

    @atom_id.setter
    def atom_id(self, atom_id: int):
        self._atom_id = atom_id

    @elem_type.setter
    def elem_type(self, elem_type: str):
        self._elem_type = elem_type

    def append_first_nearest_neighbor_list(self, index: int) -> None:
        self._first_nearest_neighbor_list.append(index)

    def append_second_nearest_neighbor_list(self, index: int) -> None:
        self._second_nearest_neighbor_list.append(index)

    def append_third_nearest_neighbor_list(self, index: int) -> None:
        self._third_nearest_neighbor_list.append(index)

    def append_fourth_nearest_neighbor_list(self, index: int) -> None:
        self._fourth_nearest_neighbor_list.append(index)

    def append_fifth_nearest_neighbor_list(self, index: int) -> None:
        self._fifth_nearest_neighbor_list.append(index)

    def append_sixth_nearest_neighbor_list(self, index: int) -> None:
        self._sixth_nearest_neighbor_list.append(index)

    def append_seventh_nearest_neighbor_list(self, index: int) -> None:
        self._seventh_nearest_neighbor_list.append(index)

    def clean_neighbors_lists(self) -> None:
        self._first_nearest_neighbor_list.clear()
        self._second_nearest_neighbor_list.clear()
        self._third_nearest_neighbor_list.clear()
        self._fourth_nearest_neighbor_list.clear()
        self._fifth_nearest_neighbor_list.clear()
        self._sixth_nearest_neighbor_list.clear()
        self._seventh_nearest_neighbor_list.clear()


def get_average_relative_position_atom(atom1: Atom, atom2: Atom) -> Atom:
    if atom1.elem_type != atom2.elem_type:
        raise RuntimeError(f"types do not match")
    if atom1.atom_id != atom2.elem_type:
        raise RuntimeError(f"atom ID do not match")
    relative_distance_vector = get_relative_distance_vector(atom1, atom2)
    atom1_relative_position = atom1.relative_position
    res_relative_position = 0.5 * relative_distance_vector + atom1_relative_position
    for i in range(3):
        while res_relative_position[i] >= 1:
            res_relative_position[i] -= 1
        while res_relative_position[i] < 0:
            res_relative_position[i] += 1

    res = atom1
    res.relative_position = res_relative_position
    return res


def get_relative_distance_vector(atom1: Atom, atom2: Atom) -> np.ndarray:
    relative_distance_vector = atom2.relative_position - atom1.relative_position
    for i in range(3):
        while relative_distance_vector[i] >= 0.5:
            relative_distance_vector[i] -= 1
        while relative_distance_vector[i] < -0.5:
            relative_distance_vector[i] += 1
    return relative_distance_vector
