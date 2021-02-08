import os
import typing
import pandas as pd
from cfg.config import Config, read_config
import ansys.vac_jump as vj
# import ansys.ground_case as bs
from bond_counting import get_encode_of_config
from tqdm import tqdm


def build_pd_file(type_category_map, path='../data'):
    element_set = set(type_category_map.keys())
    reference_config = read_config(os.path.join(path, 'raw/config0', 'start.cfg'))
    cluster_mapping_vj = vj.get_average_cluster_parameters_mapping(reference_config)
    # cluster_mapping_bs = bs.get_average_cluster_parameters_mapping(reference_config)

    with open(os.path.join(path, 'raw', 'barriers.txt'), 'r') as f:
        lines = f.readlines()

    ct = 0
    data = dict()
    for i in tqdm(range(len(lines)), desc="Loading configs ..."):
        dir_path = os.path.join(path, 'raw/config' + str(i))
        line_list = lines[i].split()

        config_start = read_config(os.path.join(dir_path, 'start.cfg'))

        jump_pair = (int(line_list[0]), int(line_list[1]))
        migration_atom = config_start.atom_list[jump_pair[1]].elem_type
        barriers = (float(line_list[2]), float(line_list[3]))
        ground_energies = (float(line_list[4]), float(line_list[5]))
        one_hot_encodes = vj.get_one_hot_encoding_list_forward_and_backward_from_map(
            config_start, jump_pair, element_set, cluster_mapping_vj)
        cluster_extension_parameters = vj.get_average_cluster_parameters_forward_and_backward_from_map(
            config_start, jump_pair, type_category_map, cluster_mapping_vj)

        # ground_encodes = bs.get_one_hot_encoding_list_map(config_start, jump_pair, element_set,
        #                                                   cluster_mapping_bs)

        config_end = read_config(os.path.join(dir_path, 'end.cfg'))
        ground_encodes = [get_encode_of_config(config_start, element_set),
                          get_encode_of_config(config_end, element_set)]

        data[ct] = [i, migration_atom, barriers[0], barriers[0] - barriers[1], 0.5 * (barriers[0] + barriers[1]),
                    ground_energies[0], ground_energies[1], one_hot_encodes[0], cluster_extension_parameters[0],
                    one_hot_encodes[1], cluster_extension_parameters[1], ground_encodes[0], ground_encodes[1]]
        ct += 1
        data[ct] = [i, migration_atom, barriers[1], barriers[1] - barriers[0], 0.5 * (barriers[0] + barriers[1]),
                    ground_energies[1], ground_energies[0], one_hot_encodes[1], cluster_extension_parameters[1],
                    one_hot_encodes[0], cluster_extension_parameters[0], ground_encodes[1], ground_encodes[0]]
        ct += 1

    df = pd.DataFrame.from_dict(data, orient='index',
                                columns=['index', 'migration_atom',
                                         'migration_barriers', 'energy_difference', 'e0', 'energy_start', 'energy_end',
                                         'one_hot_encode_forward', 'cluster_extension_forward',
                                         'one_hot_encode_backward', 'cluster_extension_backward',
                                         'energy_encode_start', 'energy_encode_end'])
    df.to_pickle(os.path.join(path, 'processed', 'linear_regression.pkl'))


if __name__ == '__main__':
    build_pd_file({'Al': 0, 'Mg': 2, 'Zn': -1})
