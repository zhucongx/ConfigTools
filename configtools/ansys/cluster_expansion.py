from configtools.cfg import *
import typing
from collections import OrderedDict
import bisect
import copy
import math


class Cluster(object):
    def __init__(self, label: int, type_list):
        self._label = label
        self._type_list = type_list
        self._type_list.sort()

    def __lt__(self, other):
        if self.size < other.size:
            return True
        if self.size > other.size:
            return False
        if self._label < other.label:
            return True
        if self._label > other.label:
            return False
        for type1, type2 in zip(self._type_list, other.type_list):
            if type1 < type2:
                return True
            if type1 > type2:
                return False
        return False

    def __eq__(self, other):
        if self.size != other.size:
            return False
        if self._label != other.label:
            return False
        for type1, type2 in zip(self._type_list, other.type_list):
            if type1 != type2:
                return False
        return True

    def __hash__(self):
        the_hash = hash(self._label) ^ hash(tuple(self._type_list))
        return the_hash

    def __repr__(self):
        return str(self._label) + "-".join(self._type_list)

    @property
    def label(self) -> int:
        return self._label

    @property
    def type_list(self) -> typing.List[str]:
        return self._type_list

    @property
    def size(self) -> int:
        return len(self._type_list)


# 0 singlet
# 1 first pair
# 2 second pair
# 3 third pair
# 4 first first first triplet
# 5 first first second triplet
# 6 first first third triplet
# 7 first second third triplet
# 8 first third third triplet
# 9 second third third triplet
# 10 third third third triplet
def creat_cluster_hashmap(type_set: typing.Set[str]) -> typing.Dict[Cluster, float]:
    cluster_hashmap: typing.Dict[Cluster, float] = dict()
    for type1 in type_set:
        cluster_hashmap[Cluster(0, [type1])] = 0
        for type2 in type_set:
            if type2 == "X":
                continue
            if type1 == "X" and type2[0] == "p":
                continue
            for i in range(1, 4):
                cluster_hashmap[Cluster(i, [type1, type2])] = 0
            for type3 in type_set:
                if type3 == "X" or type3[0] == 'p':
                    continue
                for i in range(4, 11):
                    cluster_hashmap[Cluster(i, [type1, type2, type3])] = 0
    return cluster_hashmap


def get_encode_of_config(config: Config, type_set: typing.Set[str]):
    cluster_hashmap = creat_cluster_hashmap(type_set)
    cluster_counter: typing.Dict[int, int] = dict()
    for i in range(11):
        cluster_counter[i] = 0
    for atom1 in config.atom_list:
        type1 = atom1.elem_type
        # 0 singlet
        cluster_hashmap[Cluster(0, [type1])] += 1
        cluster_counter[0] += 1
        for atom2_index in atom1.first_nearest_neighbor_list:
            atom2 = config.atom_list[atom2_index]
            type2 = atom2.elem_type
            # 1 first pair
            cluster_hashmap[Cluster(1, [type1, type2])] += 1
            cluster_counter[1] += 1
            for atom3_index in atom2.first_nearest_neighbor_list:
                type3 = config.atom_list[atom3_index].elem_type
                if atom3_index in atom1.first_nearest_neighbor_list:
                    # 4 first first first triplet
                    cluster_hashmap[Cluster(4, [type1, type2, type3])] += 1
                    cluster_counter[4] += 1
                elif atom3_index in atom1.second_nearest_neighbor_list:
                    # 5 first first second triplet
                    cluster_hashmap[Cluster(5, [type1, type2, type3])] += 1
                    cluster_counter[5] += 1
                elif atom3_index in atom1.third_nearest_neighbor_list:
                    # 6 first first third triplet
                    cluster_hashmap[Cluster(6, [type1, type2, type3])] += 1
                    cluster_counter[6] += 1
            for atom3_index in atom2.second_nearest_neighbor_list:
                type3 = config.atom_list[atom3_index].elem_type
                if atom3_index in atom1.third_nearest_neighbor_list:
                    # 7 first second third triplet
                    cluster_hashmap[Cluster(7, [type1, type2, type3])] += 1
                    cluster_counter[7] += 1
            for atom3_index in atom2.third_nearest_neighbor_list:
                type3 = config.atom_list[atom3_index].elem_type
                if atom3_index in atom1.third_nearest_neighbor_list:
                    # 8 first third third triplet
                    cluster_hashmap[Cluster(8, [type1, type2, type3])] += 1
                    cluster_counter[8] += 1
        for atom2_index in atom1.second_nearest_neighbor_list:
            atom2 = config.atom_list[atom2_index]
            type2 = atom2.elem_type
            # 2 second pair
            cluster_hashmap[Cluster(2, [type1, type2])] += 1
            cluster_counter[2] += 1
            for atom3_index in atom2.third_nearest_neighbor_list:
                type3 = config.atom_list[atom3_index].elem_type
                if atom3_index in atom1.third_nearest_neighbor_list:
                    # 9 second third third triplet
                    cluster_hashmap[Cluster(9, [type1, type2, type3])] += 1
                    cluster_counter[9] += 1
        for atom2_index in atom1.third_nearest_neighbor_list:
            atom2 = config.atom_list[atom2_index]
            type2 = atom2.elem_type
            # 3 third pair
            cluster_hashmap[Cluster(3, [type1, type2])] += 1
            cluster_counter[3] += 1
            for atom3_index in atom2.third_nearest_neighbor_list:
                type3 = config.atom_list[atom3_index].elem_type
                if atom3_index in atom1.third_nearest_neighbor_list:
                    # 10 third third third triplet
                    cluster_hashmap[Cluster(10, [type1, type2, type3])] += 1
                    cluster_counter[10] += 1
    cluster_hashmap = OrderedDict(sorted(cluster_hashmap.items(), key=lambda it: it[0]))
    res = []
    for cluster in cluster_hashmap:
        res.append(cluster_hashmap[cluster] / cluster_counter[cluster.label])
    return res


def get_encodes(config_start: Config, config_end: Config, jump_pair: typing.Tuple[int, int], type_set: typing.Set[str]):
    migration_type = config_start.atom_list[jump_pair[1]].elem_type
    _type_set = copy.copy(type_set)
    _type_set.add("X")
    _type_set.add("p"+migration_type)
    cluster_expansion_start = get_encode_of_config(config_start, _type_set)
    cluster_expansion_end = get_encode_of_config(config_end, _type_set)
    cluster_expansion_forward, cluster_expansion_backward = [], []
    for x, y in zip(cluster_expansion_start, cluster_expansion_end):
        cluster_expansion_forward.append(y - x)
        cluster_expansion_backward.append(x - y)
    config_transition = copy.copy(config_start)
    config_transition.atom_list[jump_pair[0]].elem_type = "p"+migration_type
    config_transition.atom_list[jump_pair[1]].elem_type = "p"+migration_type
    cluster_expansion_transition = get_encode_of_config(config_transition, _type_set)
    return cluster_expansion_start, cluster_expansion_end, \
           cluster_expansion_forward, cluster_expansion_backward, \
           cluster_expansion_transition


if __name__ == "__main__":
    # aa = creat_cluster_hashmap({"Al", "Mg", "Zn", "X", "pAl"})
    # aa = OrderedDict(sorted(aa.items(), key=lambda it: it[0]))
    # for a in aa:
    #     print(a, aa[a])
    # print(len(aa))
    conf_s = read_config("../../test/test_files/forward.cfg")
    conf_e = read_config("../../test/test_files/backward.cfg")
    aaa = get_encodes(conf_s, conf_e, (18, 23), {"Al", "Mg", "Zn"})
    for a in aaa:
        print(len(a))
    print(aaa[2])
