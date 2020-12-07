from cfg.config import *
from ansys.cluster_ansys import *


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # config = read_poscar("POSCAR")
    # write_poscar(config, "T")
    config = read_config("T")
    # get_symmetrically_sorted_atom_vectors(config, (82, 83))
    write_config(config, "T2", )
    print(get_relative_distance_vector(config.atom_list[82], config.atom_list[83]))
    print(get_pair_center(config, (82, 83)))
    rotation_matrix = get_pair_rotation_matrix(config, (82, 83))
    atom_list = copy.deepcopy(config.atom_list)
    print([atom.relative_position for atom in atom_list])
    print('*'*80)
    rotate_atom_vector(atom_list, rotation_matrix)
    print([atom.relative_position for atom in atom_list])

    write_config(config, "T3", )

    # write_config(config, "T2", False)
