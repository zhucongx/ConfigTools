from cfg.config import *
from ansys.vac_jump import *
from gnn.dataset import *
import bisect

if __name__ == '__main__':
    # config = read_poscar("POSCAR")
    # write_poscar(config, "T")
    config = read_config("test/test_files/test.cfg")
    write_config(config, "T2")
    cluster_mapping = get_average_cluster_parameters_mapping(config)
    _, _ = get_one_hot_encoding_list_forward_and_backward_from_map(
        config, (18, 23), {'Al', 'Mg', 'Zn'}, cluster_mapping)
    _, _ = get_average_cluster_parameters_forward_and_backward_from_map(
        config, (18, 23), {'Al': 0, 'Mg': 2, 'Zn': -1}, cluster_mapping)
    write_config(config, "T3")

    # print(get_relative_distance_vector(config.atom_list[82], config.atom_list[83]))
    # print(get_pair_center(config, (82, 83)))
    # rotation_matrix = get_pair_rotation_matrix(config, (82, 83))
    # atom_list = copy.deepcopy(config.atom_list)
    # print([atom.relative_position for atom in atom_list])
    # print('*'*80)
    # rotate_atom_vector(atom_list, rotation_matrix)
    # print([atom.relative_position for atom in atom_list])
    #
    # write_config(config, "T3", )

    # write_config(config, "T2", False)
