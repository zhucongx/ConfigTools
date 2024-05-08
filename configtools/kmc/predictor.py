import numpy as np

from configtools import cfg
from configtools.ansys import get_encode_change_of_config
import typing
import json


class Predictor(object):
    def __init__(self, json_file_name, type_set: typing.Set[str]):
        with open(json_file_name) as f:
            data = json.load(f)
        self.bond_theta = data['Bond']['theta']
        self.e0 = {'Al': 0.58, 'Zn': 0.34, 'Mg': 0.47}
        self.type_set = type_set

    def get_barrier_and_diff(self, config: cfg.Config, jump_pair: typing.Tuple[int, int]):
        element_type = config.atom_list[jump_pair[1]].elem_type
        e0 = self.e0[element_type]
        bond_change_encode = get_encode_change_of_config(config, jump_pair, self.type_set)
        de = np.dot(bond_change_encode, self.bond_theta)
        return e0 + de / 2, de
