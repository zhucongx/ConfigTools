import os
import typing

import numpy as np
import pandas as pd
from cfg.config import Config, read_config, get_config_system
import ansys.cluster_expansion as ce
import bond_counting as bc
from tqdm import tqdm


def build_pd_file(type_category_map, path="../data"):
    element_set = set(type_category_map.keys())
    reference_config = read_config(os.path.join(path, "raw/config0", "start.cfg"))
    cluster_mapping = ce.get_average_cluster_parameters_mapping(reference_config)

    df_barriers = pd.read_csv(os.path.join(path, "raw", "barriers.txt"), sep='\t')

    ct = 0
    data = dict()
    for i in tqdm(range(len(df_barriers)), desc="Loading configs ..."):
        dir_path = os.path.join(path, "raw/config" + str(i))

        config_start = read_config(os.path.join(dir_path, "start.cfg"))

        jump_pair = (df_barriers.at[i, 'VacID'], df_barriers.at[i, 'JumpID'])
        migration_atom = config_start.atom_list[jump_pair[1]].elem_type
        migration_system = get_config_system(config_start)
        barriers = (df_barriers.at[i, 'ERGforward'], df_barriers.at[i, 'ERGbackward'])
        ground_energies = (df_barriers.at[i, 'ERGstartshow'], df_barriers.at[i, 'ERGendshow'])
        force = df_barriers.at[i, 'FORCEsaddle']
        distance = df_barriers.at[i, 'DIST']
        distance_list = np.array([df_barriers.at[i, 'DISTstart'], df_barriers.at[i, 'DIST1'],
                                  df_barriers.at[i, 'DIST2'], df_barriers.at[i, 'DIST3'],
                                  df_barriers.at[i, 'DIST4'], df_barriers.at[i, 'DIST5'],
                                  df_barriers.at[i, 'DISTend']])
        distance_end = float(distance_list[-1])
        distance_list_back = distance_end - distance_list
        distance_list_back = distance_list_back[::-1]

        energy_list = np.array([df_barriers.at[i, 'ERGstart'], df_barriers.at[i, 'ERG1'],
                                df_barriers.at[i, 'ERG2'], df_barriers.at[i, 'ERG3'],
                                df_barriers.at[i, 'ERG4'], df_barriers.at[i, 'ERG5'],
                                df_barriers.at[i, 'ERGend']])
        energy_end = float(energy_list[-1])
        energy_list_back = energy_list - energy_end
        energy_list_back = energy_list_back[::-1]

        one_hot_encodes_forward = ce.get_one_hot_encoding_list_forward_and_backward_from_map(
            config_start, jump_pair, element_set, cluster_mapping)

        config_end = read_config(os.path.join(dir_path, "end.cfg"))
        ground_encode_initial = bc.get_encode_of_config(config_start, element_set)
        ground_encode_final = bc.get_encode_of_config(config_end, element_set)

        bond_change_forward = []
        bond_change_backward = []
        for x, y in zip(ground_encode_initial, ground_encode_final):
            bond_change_forward.append(y - x)
            bond_change_backward.append(x - y)

        one_hot_encodes_backward = ce.get_one_hot_encoding_list_forward_and_backward_from_map(
            config_end, jump_pair, element_set, cluster_mapping)

        data[ct] = [i, migration_atom, migration_system, barriers[0], barriers[0] - barriers[1],
                    0.5 * (barriers[0] + barriers[1]), ground_energies[0], ground_energies[1],
                    distance, distance_list[-1], force,
                    one_hot_encodes_forward[0], one_hot_encodes_backward[0],
                    ground_encode_initial, ground_encode_final, bond_change_forward, bond_change_backward,
                    distance_list, energy_list]
        ct += 1
        data[ct] = [i, migration_atom, migration_system, barriers[1], barriers[1] - barriers[0],
                    0.5 * (barriers[0] + barriers[1]), ground_energies[1], ground_energies[0],
                    distance, distance_list_back[-1], force,
                    one_hot_encodes_backward[0], one_hot_encodes_forward[0],
                    ground_encode_final, ground_encode_initial, bond_change_backward, bond_change_forward,
                    distance_list_back, energy_list_back]
        ct += 1

    print("done!!!!!")
    df = pd.DataFrame.from_dict(data, orient="index",
                                columns=["index", "migration_atom", "migration_system",
                                         "migration_barriers", "energy_difference", "e0",
                                         "energy_start", "energy_end",
                                         "distance", "min_erg_distance", "saddle_force",
                                         "one_hot_encode_forward", "one_hot_encode_backward",
                                         "energy_encode_start", "energy_encode_end", "bond_change_encode_forward",
                                         "bond_change_encode_backward", "distance_list", "energy_list"])
    df.to_pickle(os.path.join(path, "processed", "linear_regression.pkl"))


if __name__ == "__main__":
    build_pd_file({"Al": 0, "Mg": 2, "Zn": -1})
