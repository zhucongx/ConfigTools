import os

import numpy as np
import pandas as pd
import configtools.cfg as cfg
import configtools.ansys.cluster_expansion_symmetry_mm2 as ces2
import configtools.ansys.cluster_expansion_symmetry_mmm as cesm
import configtools.ansys.cluster_expansion_periodic as cep
import configtools.ansys.bond_counting as bc
from tqdm import tqdm


def build_pd_file(element_set, path, out_put_destination):
    reference_config = cfg.read_config(os.path.join(path, "config0", "start.cfg"))
    cluster_mapping_symmetry_mmm = cesm.get_average_cluster_parameters_mapping_symmetry(reference_config)
    cluster_mapping_symmetry_mm2 = ces2.get_average_cluster_parameters_mapping_symmetry(reference_config)
    cluster_mapping_periodic = cep.get_average_cluster_parameters_mapping_periodic(reference_config)
    df_barriers = pd.read_csv(os.path.join(path, "barriers.txt"), sep='\t')
    ct = 0

    data = dict()
    for i in tqdm(range(len(df_barriers)), desc="Loading configs ..."):
        dir_path = os.path.join(path, "config" + str(i))

        config_start = cfg.read_config(os.path.join(dir_path, "start.cfg"))
        config_end = cfg.read_config(os.path.join(dir_path, "end.cfg"))

        jump_pair = (df_barriers.at[i, 'VacID'], df_barriers.at[i, 'JumpID'])
        migration_atom = config_start.atom_list[jump_pair[1]].elem_type
        migration_system = cfg.get_config_system(config_start)
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

        one_hot_encodes_forward_mmm = cesm.get_one_hot_encoding_list_forward_and_backward_from_mapping(
            config_start, jump_pair, element_set, cluster_mapping_symmetry_mmm)
        one_hot_encodes_backward_mmm = cesm.get_one_hot_encoding_list_forward_and_backward_from_mapping(
            config_end, jump_pair, element_set, cluster_mapping_symmetry_mmm)
        one_hot_encodes_forward_mm2 = ces2.get_one_hot_encoding_list_forward_and_backward_from_mapping(
            config_start, jump_pair, element_set, cluster_mapping_symmetry_mm2)
        one_hot_encodes_backward_mm2 = ces2.get_one_hot_encoding_list_forward_and_backward_from_mapping(
            config_end, jump_pair, element_set, cluster_mapping_symmetry_mm2)
        bond_counting_ground_encode_start = bc.get_encode_of_config(config_start, element_set)
        bond_counting_ground_encode_end = bc.get_encode_of_config(config_end, element_set)
        bond_change_forward = []
        bond_change_backward = []
        for x, y in zip(bond_counting_ground_encode_start, bond_counting_ground_encode_end):
            bond_change_forward.append(y - x)
            bond_change_backward.append(x - y)

        cluster_expansion_ground_encode_start = cep.get_one_hot_encoding_list_from_mapping(
            config_start, element_set, cluster_mapping_periodic)
        cluster_expansion_ground_encode_end = cep.get_one_hot_encoding_list_from_mapping(
            config_end, element_set, cluster_mapping_periodic)
        cluster_expansion_change_forward = []
        cluster_expansion_change_backward = []
        for x, y in zip(cluster_expansion_ground_encode_start, cluster_expansion_ground_encode_end):
            cluster_expansion_change_forward.append(y - x)
            cluster_expansion_change_backward.append(x - y)

        data[ct] = [i, migration_atom, migration_system, barriers[0], barriers[0] - barriers[1],
                    0.5 * (barriers[0] + barriers[1]), ground_energies[0], ground_energies[1],
                    distance, distance_list[-1], force,
                    one_hot_encodes_forward_mmm[0], one_hot_encodes_backward_mmm[0],
                    one_hot_encodes_forward_mm2[0], one_hot_encodes_backward_mm2[0],
                    # bond_counting_ground_encode_start, bond_counting_ground_encode_end,
                    bond_change_forward, bond_change_backward,
                    # cluster_expansion_ground_encode_start, cluster_expansion_ground_encode_end,
                    cluster_expansion_change_forward, cluster_expansion_change_backward,
                    distance_list, energy_list]
        ct += 1
        data[ct] = [i, migration_atom, migration_system, barriers[1], barriers[1] - barriers[0],
                    0.5 * (barriers[0] + barriers[1]), ground_energies[1], ground_energies[0],
                    distance, distance_list_back[-1], force,
                    one_hot_encodes_backward_mmm[0], one_hot_encodes_forward_mmm[0],
                    one_hot_encodes_backward_mm2[0], one_hot_encodes_forward_mm2[0],
                    # bond_counting_ground_encode_end, bond_counting_ground_encode_start,
                    bond_change_backward, bond_change_forward,
                    # cluster_expansion_ground_encode_end, cluster_expansion_ground_encode_start,
                    cluster_expansion_change_backward, cluster_expansion_change_forward,
                    distance_list_back, energy_list_back]
        ct += 1

    print("done!!!!!")
    df = pd.DataFrame.from_dict(
        data, orient="index",
        columns=["index", "migration_atom", "migration_system", "migration_barriers", "energy_difference",
                 "e0", "energy_start", "energy_end",
                 "distance", "min_erg_distance", "saddle_force",
                 "one_hot_encode_forward_mmm", "one_hot_encode_backward_mmm",
                 "one_hot_encode_forward_mm2", "one_hot_encode_backward_mm2",
                 # "bond_counting_encode_start", "bond_counting_encode_end",
                 "bond_change_encode_forward", "bond_change_encode_backward",
                 # "cluster_expansion_encode_start", "cluster_expansion_encode_end",
                 "cluster_expansion_change_forward", "cluster_expansion_change_backward",
                 "distance_list", "energy_list"])
    df.to_pickle(out_put_destination, compression = 'gzip')
    
# if __name__ == "__main__":
#     build_pd_file({"Al", "Mg", "Zn"}, "../../../all_data_neb_result/", "../../data/all_compiled_data_mm2.pkl")
