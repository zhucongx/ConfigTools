from configtools.cfg import *
from collections import OrderedDict
import bisect
import copy
import math
import typing


class Cluster(object):
    def __init__(self, *types: str):
        self._type_list: typing.List[str] = list()
        for insert_type in types:
            if types == "X":
                print("type X..!!!")
            self._type_list.append(insert_type)
        self._type_list.sort()

    def __lt__(self, other):
        for type1, type2 in zip(self._type_list, other.type_list):
            if type1 < type2:
                return True
            if type1 > type2:
                return False
        return False

    def __eq__(self, other):
        for type1, type2 in zip(self._type_list, other.type_list):
            if type1 != type2:
                return False
        return True

    def __hash__(self):
        the_hash = hash(tuple(self._type_list))
        return the_hash

    def __str__(self):
        return "-".join(self._type_list)

    def __repr__(self):
        return "-".join(self._type_list)

    @property
    def type_list(self) -> typing.List[str]:
        return self._type_list

    @property
    def size(self) -> int:
        return len(self._type_list)


def count_all_cluster(config: Config, type_set: typing.Set[str]):
    singlets_count: typing.Dict[Cluster, int] = dict()
    first_pair_count: typing.Dict[Cluster, int] = dict()
    second_pair_count: typing.Dict[Cluster, int] = dict()
    third_pair_count: typing.Dict[Cluster, int] = dict()
    first_first_first_triplets_count: typing.Dict[Cluster, int] = dict()
    first_first_second_triplets_count: typing.Dict[Cluster, int] = dict()
    first_first_third_triplets_count: typing.Dict[Cluster, int] = dict()
    first_second_third_triplets_count: typing.Dict[Cluster, int] = dict()
    first_third_third_triplets_count: typing.Dict[Cluster, int] = dict()
    second_third_third_triplets_count: typing.Dict[Cluster, int] = dict()
    third_third_third_triplets_count: typing.Dict[Cluster, int] = dict()
    # initialize dict
    for type1 in type_set:
        singlet = Cluster(type1)
        singlets_count[singlet] = 0
        for type2 in type_set:
            pair = Cluster(type1, type2)
            first_pair_count[pair] = 0
            second_pair_count[pair] = 0
            third_pair_count[pair] = 0
            for type3 in type_set:
                triplet = Cluster(type1, type2, type3)
                first_first_first_triplets_count[triplet] = 0
                first_first_second_triplets_count[triplet] = 0
                first_first_third_triplets_count[triplet] = 0
                first_second_third_triplets_count[triplet] = 0
                first_third_third_triplets_count[triplet] = 0
                second_third_third_triplets_count[triplet] = 0
                third_third_third_triplets_count[triplet] = 0

    for atom1 in config.atom_list:
        type1 = atom1.elem_type
        if type1 == "X":
            continue
        singlets_count[Cluster(type1)] += 1
        for atom2_index in atom1.first_nearest_neighbor_list:
            atom2 = config.atom_list[atom2_index]
            type2 = atom2.elem_type
            if type2 == "X":
                continue
            first_pair_count[Cluster(type1, type2)] += 1
            for atom3_index in atom2.first_nearest_neighbor_list:
                type3 = config.atom_list[atom3_index].elem_type
                if type3 == "X":
                    continue
                if atom3_index in atom1.first_nearest_neighbor_list:
                    first_first_first_triplets_count[Cluster(type1, type2, type3)] += 1
                if atom3_index in atom1.second_nearest_neighbor_list:
                    first_first_second_triplets_count[Cluster(type1, type2, type3)] += 1
                if atom3_index in atom1.third_nearest_neighbor_list:
                    first_first_third_triplets_count[Cluster(type1, type2, type3)] += 1
            for atom3_index in atom2.second_nearest_neighbor_list:
                type3 = config.atom_list[atom3_index].elem_type
                if type3 == "X":
                    continue
                if atom3_index in atom1.third_nearest_neighbor_list:
                    first_second_third_triplets_count[Cluster(type1, type2, type3)] += 1
            for atom3_index in atom2.third_nearest_neighbor_list:
                type3 = config.atom_list[atom3_index].elem_type
                if type3 == "X":
                    continue
                if atom3_index in atom1.third_nearest_neighbor_list:
                    first_third_third_triplets_count[Cluster(type1, type2, type3)] += 1
        for atom2_index in atom1.second_nearest_neighbor_list:
            atom2 = config.atom_list[atom2_index]
            type2 = atom2.elem_type
            if type2 == "X":
                continue
            second_pair_count[Cluster(type1, type2)] += 1
            for atom3_index in atom2.third_nearest_neighbor_list:
                type3 = config.atom_list[atom3_index].elem_type
                if type3 == "X":
                    continue
                if atom3_index in atom1.third_nearest_neighbor_list:
                    second_third_third_triplets_count[Cluster(type1, type2, type3)] += 1
        for atom2_index in atom1.third_nearest_neighbor_list:
            atom2 = config.atom_list[atom2_index]
            type2 = atom2.elem_type
            if type2 == "X":
                continue
            third_pair_count[Cluster(type1, type2)] += 1
            for atom3_index in atom2.third_nearest_neighbor_list:
                type3 = config.atom_list[atom3_index].elem_type
                if type3 == "X":
                    continue
                if atom3_index in atom1.third_nearest_neighbor_list:
                    third_third_third_triplets_count[Cluster(type1, type2, type3)] += 1

    for bond in first_pair_count.keys():
        first_pair_count[bond] /= 2
    for bond in second_pair_count.keys():
        second_pair_count[bond] /= 2
    for bond in third_pair_count.keys():
        third_pair_count[bond] /= 2

    for bond in first_first_first_triplets_count.keys():
        first_first_first_triplets_count[bond] /= 6
    for bond in first_first_second_triplets_count.keys():
        first_first_second_triplets_count[bond] /= 6
    for bond in first_first_third_triplets_count.keys():
        first_first_third_triplets_count[bond] /= 6
    for bond in first_second_third_triplets_count.keys():
        first_second_third_triplets_count[bond] /= 6
    for bond in first_third_third_triplets_count.keys():
        first_third_third_triplets_count[bond] /= 6
    for bond in second_third_third_triplets_count.keys():
        second_third_third_triplets_count[bond] /= 6
    for bond in third_third_third_triplets_count.keys():
        third_third_third_triplets_count[bond] /= 6
    return OrderedDict(sorted(singlets_count.items(), key=lambda it: it[0])), \
           OrderedDict(sorted(first_pair_count.items(), key=lambda it: it[0])), \
           OrderedDict(sorted(second_pair_count.items(), key=lambda it: it[0])), \
           OrderedDict(sorted(third_pair_count.items(), key=lambda it: it[0])), \
           OrderedDict(sorted(first_first_first_triplets_count.items(), key=lambda it: it[0])), \
           OrderedDict(sorted(first_first_second_triplets_count.items(), key=lambda it: it[0])), \
           OrderedDict(sorted(first_first_third_triplets_count.items(), key=lambda it: it[0])), \
           OrderedDict(sorted(first_second_third_triplets_count.items(), key=lambda it: it[0])), \
           OrderedDict(sorted(first_third_third_triplets_count.items(), key=lambda it: it[0])), \
           OrderedDict(sorted(second_third_third_triplets_count.items(), key=lambda it: it[0])), \
           OrderedDict(sorted(third_third_third_triplets_count.items(), key=lambda it: it[0]))


def get_encode_of_config(config: Config, type_set: typing.Set[str]) -> typing.List[float]:
    res = []
    singlet, first_pair, second_pair, third_pair, first_first_first_triplet, first_first_second_triplet, \
    first_first_third_triplet, first_second_third_triplet, first_third_third_triplet, second_third_third, \
    third_third_third_triplet = count_all_cluster(config, type_set)

    for bond in singlet.keys():
        res.append(singlet[bond])
    for bond in first_pair.keys():
        res.append(first_pair[bond])
    for bond in second_pair.keys():
        res.append(second_pair[bond])
    for bond in third_pair.keys():
        res.append(third_pair[bond])
    for bond in first_first_first_triplet.keys():
        res.append(first_first_first_triplet[bond])
    for bond in first_first_second_triplet.keys():
        res.append(first_first_second_triplet[bond])
    for bond in first_first_third_triplet.keys():
        res.append(first_first_third_triplet[bond])
    for bond in first_second_third_triplet.keys():
        res.append(first_second_third_triplet[bond])
    for bond in first_third_third_triplet.keys():
        res.append(first_third_third_triplet[bond])
    for bond in second_third_third.keys():
        res.append(second_third_third[bond])
    for bond in third_third_third_triplet.keys():
        res.append(third_third_third_triplet[bond])
    return res


if __name__ == "__main__":
    config = read_config("../../test/test_files/forward.cfg")
    # atom1 = config.atom_list[0]
    # res = []
    # for atom2 in config.atom_list:
    #     relative_distance_vector = get_relative_distance_vector(atom1, atom2)
    #     absolute_distance_vector = relative_distance_vector.dot(config.basis)
    #     res.append(np.sqrt(np.inner(absolute_distance_vector, absolute_distance_vector)))
    # res = sorted(res)
    # for i in res:
    #     print(i)
    a, b, c, d, e, f, g, h, i, j, k = count_all_cluster(config, {"Al", "Mg", "Zn"})

    print(a)
    print(b)
    print(c)
    print(d)
    print(e)
    print(f)
    print(g)
    print(h)
    print(i)
    print(j)
    print(k)
    jj = get_encode_of_config(config, {"Al", "Mg", "Zn"})
    config = read_config("../../test/test_files/backward.cfg")
    kk = get_encode_of_config(config, {"Al", "Mg", "Zn"})
    ii = []
    for j, k in zip(jj, kk):
        ii.append(k - j)
    print(ii)
