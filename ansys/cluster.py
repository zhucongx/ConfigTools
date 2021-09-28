from cfg.config import Atom
import copy
import typing


class Cluster(object):
    def __init__(self, *atoms: Atom):
        self._atom_list: typing.List[Atom] = list()
        for insert_atom in atoms:
            if insert_atom.elem_type == "X":
                print("type X..!!!")
            self._atom_list.append(copy.deepcopy(insert_atom))

        self._atom_list.sort(key=lambda sort_atom: sort_atom.atom_id)

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
    def size(self) ->int:
        return len(self._atom_list)

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

    num_triplets = len(type_set) ** 3
    counter = 0
    for element1 in sorted_type_set:
        for element2 in sorted_type_set:
            for element3 in sorted_type_set:
                element_encode = [0.] * num_triplets
                element_encode[counter] = 1.
                encode_dict[element1 + element2 + element3] = element_encode
                counter += 1

    num_quadruplets = len(type_set) ** 4
    counter = 0
    for element1 in sorted_type_set:
        for element2 in sorted_type_set:
            for element3 in sorted_type_set:
                for element4 in sorted_type_set:
                    element_encode = [0.] * num_quadruplets
                    element_encode[counter] = 1.
                    encode_dict[element1 + element2 + element3 + element4] = element_encode
                    counter += 1

    return encode_dict


def element_wise_add_second_to_first(first_list: typing.List[float], second_list: typing.List[float]):
    if len(first_list) != len(second_list):
        raise RuntimeError("Size mismatch")
    for i in range(len(first_list)):
        first_list[i] += second_list[i]


def element_wise_divide_float_from_list(float_list: typing.List[float], divisor: float):
    for i in range(len(float_list)):
        float_list[i] /= divisor