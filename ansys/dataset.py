import os
import typing
import pandas as pd
from cfg.config import Config, read_config
from ansys.vac_jump import get_average_cluster_parameters_mapping, \
    get_one_hot_encoding_list_forward_and_backward_from_map, \
    get_average_cluster_parameters_forward_and_backward_from_map
from tqdm import tqdm


def build_pd_file(type_category_map, path='../data'):
    element_set = set(type_category_map.keys())

    cluster_mapping = get_average_cluster_parameters_mapping(
        read_config(os.path.join(path, 'raw/config0', 'start.cfg')))

    with open(os.path.join(path, 'raw', 'barriers.txt'), 'r') as f:
        lines = f.readlines()

    ct = 0
    data = dict()
    for i in tqdm(range(len(lines)), desc="Loading configs ..."):
        dir_path = os.path.join(path, 'raw/config' + str(i))
        line_list = lines[i].split()

        config = read_config(os.path.join(dir_path, 'start.cfg'))
        jump_pair = (int(line_list[0]), int(line_list[1]))
        migration_atom = config.atom_list[jump_pair[1]].elem_type
        barriers = (float(line_list[2]), float(line_list[3]))
        one_hot_encodes = get_one_hot_encoding_list_forward_and_backward_from_map(config, jump_pair,
                                                                                  element_set,
                                                                                  cluster_mapping)
        cluster_extension_parameters = get_average_cluster_parameters_forward_and_backward_from_map(config, jump_pair,
                                                                                                    type_category_map,
                                                                                                    cluster_mapping)
        data[ct] = [i, migration_atom, barriers[0], barriers[0] - barriers[1], one_hot_encodes[0],
                    cluster_extension_parameters[0], one_hot_encodes[1], cluster_extension_parameters[1]]
        ct += 1
        data[ct] = [i, migration_atom, barriers[1], barriers[1] - barriers[0], one_hot_encodes[1],
                    cluster_extension_parameters[1], one_hot_encodes[0], cluster_extension_parameters[0]]
        ct += 1

    df = pd.DataFrame.from_dict(data, orient='index',
                                columns=['index', 'migration_atom',
                                         'migration_barriers', 'energy_difference',
                                         'one_hot_encode_forward', 'cluster_extension_forward',
                                         'one_hot_encode_backward', 'cluster_extension_backward'])
    df.to_pickle(os.path.join(path, 'processed', 'linear_regression.pkl'))


if __name__ == '__main__':
    build_pd_file({'Al': 0, 'Mg': 2, 'Zn': -1})
