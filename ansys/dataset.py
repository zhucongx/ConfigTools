import os
import typing
import pandas as pd
from cfg.config import Config, read_config
import ansys.cluster_expansion as ce
import bond_counting as bc
from tqdm import tqdm


def build_pd_file(type_category_map, path="../data"):
    element_set = set(type_category_map.keys())
    reference_config = read_config(os.path.join(path, "raw/config0", "start.cfg"))
    cluster_mapping = ce.get_average_cluster_parameters_mapping(reference_config)

    with open(os.path.join(path, "raw", "barriers.txt"), "r") as f:
        lines = f.readlines()

    ct = 0
    data = dict()
    for i in tqdm(range(len(lines)), desc="Loading configs ..."):
        dir_path = os.path.join(path, "raw/config" + str(i))
        line_list = lines[i].split()

        config_start = read_config(os.path.join(dir_path, "start.cfg"))

        jump_pair = (int(line_list[0]), int(line_list[1]))
        migration_atom = config_start.atom_list[jump_pair[1]].elem_type
        barriers = (float(line_list[2]), float(line_list[3]))

        ground_energies = (float(line_list[4]), float(line_list[5]))
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

        data[ct] = [i, migration_atom, barriers[0], barriers[0] - barriers[1], 0.5 * (barriers[0] + barriers[1]),
                    ground_energies[0], ground_energies[1], one_hot_encodes_forward[0], one_hot_encodes_backward[0],
                    ground_encode_initial, ground_encode_final, bond_change_forward, bond_change_backward]
        ct += 1
        data[ct] = [i, migration_atom, barriers[1], barriers[1] - barriers[0], 0.5 * (barriers[0] + barriers[1]),
                    ground_energies[1], ground_energies[0], one_hot_encodes_backward[0], one_hot_encodes_forward[0],
                    ground_encode_final, ground_encode_initial, bond_change_backward, bond_change_forward]
        ct += 1

    df = pd.DataFrame.from_dict(data, orient="index",
                                columns=["index", "migration_atom",
                                         "migration_barriers", "energy_difference", "e0", "energy_start", "energy_end",
                                         "one_hot_encode_forward", "one_hot_encode_backward",
                                         "energy_encode_start", "energy_encode_end",
                                         "bond_change_encode_forward", "bond_change_encode_backward"])
    df.to_pickle(os.path.join(path, "processed", "linear_regression.pkl"))


if __name__ == "__main__":
    build_pd_file({"Al": 0, "Mg": 2, "Zn": -1})
