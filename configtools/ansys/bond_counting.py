from configtools.cfg import *
from collections import OrderedDict
import bisect
import copy
import math
import typing


class Bond(object):
    def __init__(self, type1: str, type2: str):
        if type1 < type2:
            self.type1, self.type2 = type1, type2
        else:
            self.type1, self.type2 = type2, type1

    def __lt__(self, other):
        if self.type1 < other.type1:
            return True
        if self.type1 > other.type1:
            return False
        return self.type2 < other.type2

    def __eq__(self, other):
        if self.type1 == other.type1 and self.type2 == other.type2:
            return True
        return False

    def __hash__(self):
        return hash(self.type1) ^ hash(self.type2)

    def __str__(self):
        return "-".join((self.type1, self.type2))

    def __repr__(self):
        return "-".join((self.type1, self.type2))


def count_all_bond(config: Config, type_set: typing.Set[str]):
    first_bond_count: typing.Dict[Bond, int] = dict()
    second_bond_count: typing.Dict[Bond, int] = dict()
    third_bond_count: typing.Dict[Bond, int] = dict()
    fourth_bond_count: typing.Dict[Bond, int] = dict()
    fifth_bond_count: typing.Dict[Bond, int] = dict()
    sixth_bond_count: typing.Dict[Bond, int] = dict()
    seventh_bond_count: typing.Dict[Bond, int] = dict()
    # initialize dict
    for type1 in type_set:
        for type2 in type_set:
            bond = Bond(type1, type2)
            first_bond_count[bond] = 0
            second_bond_count[bond] = 0
            third_bond_count[bond] = 0
            fourth_bond_count[bond] = 0
            fifth_bond_count[bond] = 0
            sixth_bond_count[bond] = 0
            seventh_bond_count[bond] = 0

    for atom in config.atom_list:
        type1 = atom.elem_type
        if type1 == "X":
            continue
        for index in atom.first_nearest_neighbor_list:
            type2 = config.atom_list[index].elem_type
            if type2 == "X":
                continue
            bond = Bond(type1, type2)
            first_bond_count[bond] += 1
        for index in atom.second_nearest_neighbor_list:
            type2 = config.atom_list[index].elem_type
            if type2 == "X":
                continue
            bond = Bond(type1, type2)
            second_bond_count[bond] += 1
        for index in atom.third_nearest_neighbor_list:
            type2 = config.atom_list[index].elem_type
            if type2 == "X":
                continue
            bond = Bond(type1, type2)
            third_bond_count[bond] += 1
        for index in atom.fourth_nearest_neighbor_list:
            type2 = config.atom_list[index].elem_type
            if type2 == "X":
                continue
            bond = Bond(type1, type2)
            fourth_bond_count[bond] += 1
        for index in atom.fifth_nearest_neighbor_list:
            type2 = config.atom_list[index].elem_type
            if type2 == "X":
                continue
            bond = Bond(type1, type2)
            fifth_bond_count[bond] += 1
        for index in atom.sixth_nearest_neighbor_list:
            type2 = config.atom_list[index].elem_type
            if type2 == "X":
                continue
            bond = Bond(type1, type2)
            sixth_bond_count[bond] += 1
        for index in atom.seventh_nearest_neighbor_list:
            type2 = config.atom_list[index].elem_type
            if type2 == "X":
                continue
            bond = Bond(type1, type2)
            seventh_bond_count[bond] += 1

    for bond in first_bond_count.keys():
        first_bond_count[bond] /= 2
    for bond in second_bond_count.keys():
        second_bond_count[bond] /= 2
    for bond in third_bond_count.keys():
        third_bond_count[bond] /= 2
    for bond in fourth_bond_count.keys():
        fourth_bond_count[bond] /= 2
    for bond in fifth_bond_count.keys():
        fifth_bond_count[bond] /= 2
    for bond in sixth_bond_count.keys():
        sixth_bond_count[bond] /= 2
    for bond in seventh_bond_count.keys():
        seventh_bond_count[bond] /= 2

    return OrderedDict(sorted(first_bond_count.items(), key=lambda it: it[0])), \
           OrderedDict(sorted(second_bond_count.items(), key=lambda it: it[0])), \
           OrderedDict(sorted(third_bond_count.items(), key=lambda it: it[0])), \
           OrderedDict(sorted(fourth_bond_count.items(), key=lambda it: it[0])), \
           OrderedDict(sorted(fifth_bond_count.items(), key=lambda it: it[0])), \
           OrderedDict(sorted(sixth_bond_count.items(), key=lambda it: it[0])), \
           OrderedDict(sorted(seventh_bond_count.items(), key=lambda it: it[0]))


def get_encode_of_config(config: Config, type_set: typing.Set[str]):
    res = []
    first, second, third, fourth, fifth, sixth, seventh = count_all_bond(config, type_set)
    for bond in first.keys():
        res.append(first[bond])
    for bond in second.keys():
        res.append(second[bond])
    for bond in third.keys():
        res.append(third[bond])
    for bond in fourth.keys():
        res.append(fourth[bond])
    for bond in fifth.keys():
        res.append(fifth[bond])
    for bond in sixth.keys():
        res.append(sixth[bond])
    for bond in seventh.keys():
        res.append(seventh[bond])
    return res


def count_bond_change(config: Config, jump_pair: typing.Tuple[int, int], type_set: typing.Set[str]) -> typing.List[int]:
    first_bond_count: typing.Dict[Bond, int] = dict()
    second_bond_count: typing.Dict[Bond, int] = dict()
    third_bond_count: typing.Dict[Bond, int] = dict()
    fourth_bond_count: typing.Dict[Bond, int] = dict()
    fifth_bond_count: typing.Dict[Bond, int] = dict()
    sixth_bond_count: typing.Dict[Bond, int] = dict()
    seventh_bond_count: typing.Dict[Bond, int] = dict()
    # initialize dict
    for type1 in type_set:
        for type2 in type_set:
            bond = Bond(type1, type2)
            first_bond_count[bond] = 0
            second_bond_count[bond] = 0
            third_bond_count[bond] = 0
            fourth_bond_count[bond] = 0
            fifth_bond_count[bond] = 0
            sixth_bond_count[bond] = 0
            seventh_bond_count[bond] = 0

    atom_list = config.atom_list
    type1 = atom_list[jump_pair[1]].elem_type
    # plus new bonds
    for atom2_id in atom_list[jump_pair[0]].first_nearest_neighbor_list:
        if atom2_id == jump_pair[1]:
            continue
        first_bond_count[Bond(type1, atom_list[atom2_id].elem_type)] += 1
    for atom2_id in atom_list[jump_pair[0]].second_nearest_neighbor_list:
        second_bond_count[Bond(type1, atom_list[atom2_id].elem_type)] += 1
    for atom2_id in atom_list[jump_pair[0]].third_nearest_neighbor_list:
        third_bond_count[Bond(type1, atom_list[atom2_id].elem_type)] += 1
    for atom2_id in atom_list[jump_pair[0]].fourth_nearest_neighbor_list:
        fourth_bond_count[Bond(type1, atom_list[atom2_id].elem_type)] += 1
    for atom2_id in atom_list[jump_pair[0]].fifth_nearest_neighbor_list:
        fifth_bond_count[Bond(type1, atom_list[atom2_id].elem_type)] += 1
    for atom2_id in atom_list[jump_pair[0]].sixth_nearest_neighbor_list:
        sixth_bond_count[Bond(type1, atom_list[atom2_id].elem_type)] += 1
    for atom2_id in atom_list[jump_pair[0]].seventh_nearest_neighbor_list:
        seventh_bond_count[Bond(type1, atom_list[atom2_id].elem_type)] += 1
    # minus new bonds
    for atom2_id in atom_list[jump_pair[1]].first_nearest_neighbor_list:
        if atom2_id == jump_pair[0]:
            continue
        first_bond_count[Bond(type1, atom_list[atom2_id].elem_type)] -= 1
    for atom2_id in atom_list[jump_pair[1]].second_nearest_neighbor_list:
        second_bond_count[Bond(type1, atom_list[atom2_id].elem_type)] -= 1
    for atom2_id in atom_list[jump_pair[1]].third_nearest_neighbor_list:
        third_bond_count[Bond(type1, atom_list[atom2_id].elem_type)] -= 1
    for atom2_id in atom_list[jump_pair[1]].fourth_nearest_neighbor_list:
        fourth_bond_count[Bond(type1, atom_list[atom2_id].elem_type)] -= 1
    for atom2_id in atom_list[jump_pair[1]].fifth_nearest_neighbor_list:
        fifth_bond_count[Bond(type1, atom_list[atom2_id].elem_type)] -= 1
    for atom2_id in atom_list[jump_pair[1]].sixth_nearest_neighbor_list:
        sixth_bond_count[Bond(type1, atom_list[atom2_id].elem_type)] -= 1
    for atom2_id in atom_list[jump_pair[1]].seventh_nearest_neighbor_list:
        seventh_bond_count[Bond(type1, atom_list[atom2_id].elem_type)] -= 1

    return OrderedDict(sorted(first_bond_count.items(), key=lambda it: it[0])), \
           OrderedDict(sorted(second_bond_count.items(), key=lambda it: it[0])), \
           OrderedDict(sorted(third_bond_count.items(), key=lambda it: it[0])), \
           OrderedDict(sorted(fourth_bond_count.items(), key=lambda it: it[0])), \
           OrderedDict(sorted(fifth_bond_count.items(), key=lambda it: it[0])), \
           OrderedDict(sorted(sixth_bond_count.items(), key=lambda it: it[0])), \
           OrderedDict(sorted(seventh_bond_count.items(), key=lambda it: it[0]))


def get_encode_change_of_config(config: Config, jump_pair: typing.Tuple[int, int],
                                type_set: typing.Set[str]) -> typing.List[int]:
    res = []
    first, second, third, fourth, fifth, sixth, seventh = count_bond_change(config, jump_pair, type_set)
    for bond in first.keys():
        res.append(first[bond])
    for bond in second.keys():
        res.append(second[bond])
    for bond in third.keys():
        res.append(third[bond])
    for bond in fourth.keys():
        res.append(fourth[bond])
    for bond in fifth.keys():
        res.append(fifth[bond])
    for bond in sixth.keys():
        res.append(sixth[bond])
    for bond in seventh.keys():
        res.append(seventh[bond])
    return res


if __name__ == "__main__":
    # atom1 = config.atom_list[0]
    # res = []
    # for atom2 in config.atom_list:
    #     fractional_distance_vector = get_fractional_distance_vector(atom1, atom2)
    #     absolute_distance_vector = fractional_distance_vector.dot(config.basis)
    #     res.append(np.sqrt(np.inner(absolute_distance_vector, absolute_distance_vector)))
    # res = sorted(res)
    # for i in res:
    #     print(i)
    v1 = get_encode_of_config(read_config("../../test/test_files/forward.cfg"), {"Al", "Mg", "Zn"})
    v2 = get_encode_of_config(read_config("../../test/test_files/backward.cfg"), {"Al", "Mg", "Zn"})
    vc = []
    for i, j in zip(v1, v2):
        vc.append(j - i)
    print(vc)
    print(get_encode_change_of_config(read_config("../../test/test_files/forward.cfg"), (18, 23), {"Al", "Mg", "Zn"}))
