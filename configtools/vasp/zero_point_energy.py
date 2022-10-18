import typing
import numpy as np
import configtools.cfg as cfg


# def prepare_zpe_poscar(contcar_filename: str,
#                        poscar_filename: str,
#                        config_filename: str,
#                        out_filename: str,
#                        jump_atom_index: int,
#                        poscar_jump_atom_index: int) -> None:
#     contcar = cfg.read_poscar(contcar_filename, False)
#     poscar = cfg.read_poscar(poscar_filename, False)
#     config = cfg.read_config(config_filename, True)
#     # get jump atom index in poscar which should be same as that in contcar
#     vac_index = cfg.get_vacancy_index(config)
#     contcar_near_neighbors_hashset: typing.Set[int] = set()
#     # jump_atom
#     contcar_near_neighbors_hashset.add(poscar_jump_atom_index)
#     _write_selective_poscar(contcar, contcar_near_neighbors_hashset, out_filename)
#     # first neighbors
#     config_near_neighbors_hashset: typing.Set[int] = set()
#     for atom_index in config.atom_list[vac_index].first_nearest_neighbor_list + \
#                       config.atom_list[jump_atom_index].first_nearest_neighbor_list:
#         config_near_neighbors_hashset.add(atom_index)
#     config_near_neighbors_hashset.discard(vac_index)
#     for atom_index in config_near_neighbors_hashset:
#         poscar_atom_index = None
#         min_fractional_distance = 10
#         for atom in poscar.atom_list:
#             fractional_distance_vector = cfg.get_fractional_distance_vector(atom, config.atom_list[atom_index])
#             fractional_distance = np.inner(fractional_distance_vector, fractional_distance_vector)
#             if fractional_distance < min_fractional_distance:
#                 min_fractional_distance = fractional_distance
#                 poscar_atom_index = atom.atom_id
#         contcar_near_neighbors_hashset.add(poscar_atom_index)
#     _write_selective_poscar(contcar, contcar_near_neighbors_hashset, out_filename + '1')
#     # second neighbors
#     config_near_neighbors_hashset: typing.Set[int] = set()
#     for atom_index in config.atom_list[vac_index].second_nearest_neighbor_list + \
#                       config.atom_list[jump_atom_index].second_nearest_neighbor_list:
#         config_near_neighbors_hashset.add(atom_index)
#     config_near_neighbors_hashset.discard(vac_index)
#     for atom_index in config_near_neighbors_hashset:
#         poscar_atom_index = None
#         min_fractional_distance = 10
#         for atom in poscar.atom_list:
#             fractional_distance_vector = cfg.get_fractional_distance_vector(atom, config.atom_list[atom_index])
#             fractional_distance = np.inner(fractional_distance_vector, fractional_distance_vector)
#             if fractional_distance < min_fractional_distance:
#                 min_fractional_distance = fractional_distance
#                 poscar_atom_index = atom.atom_id
#         contcar_near_neighbors_hashset.add(poscar_atom_index)
#     _write_selective_poscar(contcar, contcar_near_neighbors_hashset, out_filename + '2')
#     # third neighbors
#     config_near_neighbors_hashset: typing.Set[int] = set()
#     for atom_index in config.atom_list[vac_index].third_nearest_neighbor_list + \
#                       config.atom_list[jump_atom_index].third_nearest_neighbor_list:
#         config_near_neighbors_hashset.add(atom_index)
#     config_near_neighbors_hashset.discard(vac_index)
#     for atom_index in config_near_neighbors_hashset:
#         poscar_atom_index = None
#         min_fractional_distance = 10
#         for atom in poscar.atom_list:
#             fractional_distance_vector = cfg.get_fractional_distance_vector(atom, config.atom_list[atom_index])
#             fractional_distance = np.inner(fractional_distance_vector, fractional_distance_vector)
#             if fractional_distance < min_fractional_distance:
#                 min_fractional_distance = fractional_distance
#                 poscar_atom_index = atom.atom_id
#         contcar_near_neighbors_hashset.add(poscar_atom_index)
#     _write_selective_poscar(contcar, contcar_near_neighbors_hashset, out_filename + '3')
#     # all
#     _write_all_poscar(contcar, out_filename + 'all')


def prepare_frequency_poscar(initial_config_filename: str,
                             final_config_filename: str,
                             initial_poscar_filename: str,
                             final_poscar_filename: str,
                             initial_contcar_filename: str,
                             final_contcar_filename: str,
                             transition_contcar_filename: str,
                             initial_out_filename: str,
                             final_out_filename: str,
                             transition_out_filename: str) -> None:
    initial_config = cfg.read_config(initial_config_filename, True)
    final_config = cfg.read_config(final_config_filename, True)
    initial_poscar = cfg.read_poscar(initial_poscar_filename, False)
    final_poscar = cfg.read_poscar(final_poscar_filename, False)
    initial_contcar = cfg.read_poscar(initial_contcar_filename, False)
    final_contcar = cfg.read_poscar(final_contcar_filename, False)
    transition_contcar = cfg.read_poscar(transition_contcar_filename, False)

    contcar_near_neighbors_hashset: typing.Set[int] = set()
    poscar_jump_atom_index = cfg.find_jump_id_from_poscar(initial_poscar, final_poscar)
    # jump_atom
    contcar_near_neighbors_hashset.add(poscar_jump_atom_index)
    _write_selective_poscar(initial_contcar, contcar_near_neighbors_hashset, initial_out_filename)
    _write_selective_poscar(final_contcar, contcar_near_neighbors_hashset, final_out_filename)
    _write_selective_poscar(transition_contcar, contcar_near_neighbors_hashset, transition_out_filename)

    # get jump atom index in poscar which should be same as that in contcar
    vac_index, jump_atom_index = cfg.find_jump_pair_from_cfg(initial_config, final_config)
    # first neighbors
    config_near_neighbors_hashset: typing.Set[int] = set()
    for atom_index in initial_config.atom_list[vac_index].first_nearest_neighbor_list + \
                      initial_config.atom_list[jump_atom_index].first_nearest_neighbor_list:
        config_near_neighbors_hashset.add(atom_index)
    config_near_neighbors_hashset.discard(vac_index)
    for atom_index in config_near_neighbors_hashset:
        poscar_atom_index = None
        min_fractional_distance = 10
        for atom in initial_poscar.atom_list:
            fractional_distance_vector = cfg.get_fractional_distance_vector(atom, initial_config.atom_list[atom_index])
            fractional_distance = np.inner(fractional_distance_vector, fractional_distance_vector)
            if fractional_distance < min_fractional_distance:
                min_fractional_distance = fractional_distance
                poscar_atom_index = atom.atom_id
        contcar_near_neighbors_hashset.add(poscar_atom_index)
    _write_selective_poscar(initial_contcar, contcar_near_neighbors_hashset, initial_out_filename + '1')
    _write_selective_poscar(final_contcar, contcar_near_neighbors_hashset, final_out_filename + '1')
    _write_selective_poscar(transition_contcar, contcar_near_neighbors_hashset, transition_out_filename + '1')
    # second neighbors
    config_near_neighbors_hashset: typing.Set[int] = set()
    for atom_index in initial_config.atom_list[vac_index].second_nearest_neighbor_list + \
                      initial_config.atom_list[jump_atom_index].second_nearest_neighbor_list:
        config_near_neighbors_hashset.add(atom_index)
    config_near_neighbors_hashset.discard(vac_index)
    for atom_index in config_near_neighbors_hashset:
        poscar_atom_index = None
        min_fractional_distance = 10
        for atom in initial_poscar.atom_list:
            fractional_distance_vector = cfg.get_fractional_distance_vector(atom, initial_config.atom_list[atom_index])
            fractional_distance = np.inner(fractional_distance_vector, fractional_distance_vector)
            if fractional_distance < min_fractional_distance:
                min_fractional_distance = fractional_distance
                poscar_atom_index = atom.atom_id
        contcar_near_neighbors_hashset.add(poscar_atom_index)
    _write_selective_poscar(initial_contcar, contcar_near_neighbors_hashset, initial_out_filename + '2')
    _write_selective_poscar(final_contcar, contcar_near_neighbors_hashset, final_out_filename + '2')
    _write_selective_poscar(transition_contcar, contcar_near_neighbors_hashset, transition_out_filename + '2')
    # third neighbors
    config_near_neighbors_hashset: typing.Set[int] = set()
    for atom_index in initial_config.atom_list[vac_index].third_nearest_neighbor_list + \
                      initial_config.atom_list[jump_atom_index].third_nearest_neighbor_list:
        config_near_neighbors_hashset.add(atom_index)
    config_near_neighbors_hashset.discard(vac_index)
    for atom_index in config_near_neighbors_hashset:
        poscar_atom_index = None
        min_fractional_distance = 10
        for atom in initial_poscar.atom_list:
            fractional_distance_vector = cfg.get_fractional_distance_vector(atom, initial_config.atom_list[atom_index])
            fractional_distance = np.inner(fractional_distance_vector, fractional_distance_vector)
            if fractional_distance < min_fractional_distance:
                min_fractional_distance = fractional_distance
                poscar_atom_index = atom.atom_id
        contcar_near_neighbors_hashset.add(poscar_atom_index)
    _write_selective_poscar(initial_contcar, contcar_near_neighbors_hashset, initial_out_filename + '3')
    _write_selective_poscar(final_contcar, contcar_near_neighbors_hashset, final_out_filename + '3')
    _write_selective_poscar(transition_contcar, contcar_near_neighbors_hashset, transition_out_filename + '3')
    # all
    _write_all_poscar(initial_contcar, initial_out_filename + 'all')
    _write_all_poscar(final_contcar, final_out_filename + 'all')
    _write_all_poscar(transition_contcar, transition_out_filename + 'all')

def _write_all_poscar(contcar: cfg.Config, out_filename: str) -> None:
    content = "#comment\n1.0\n"
    for basis_row in contcar.basis:
        for base in basis_row:
            content += f"{base} "
        content += "\n"
    element_list_map = contcar.get_element_list_map()

    element_str = ""
    count_str = ""
    for element, element_list in element_list_map.items():
        if element == "X":
            continue
        element_str += element + " "
        count_str += str(len(element_list)) + " "
    content += element_str + "\n" + count_str + "\n"
    content += "selective\n"
    content += "Direct\n"
    for element, element_list in element_list_map.items():
        if element == "X":
            continue
        for index in element_list:
            content += np.array2string(contcar.atom_list[int(index)].fractional_position,
                                       formatter={"float_kind": lambda x: "%.16f" % x})[1:-1] + " T T T" + "\n"

    with open(out_filename, "w") as f:
        f.write(content)


def _write_selective_poscar(contcar: cfg.Config, atom_index_list: typing.Set[int], out_filename: str) -> None:
    content = "#comment\n1.0\n"
    for basis_row in contcar.basis:
        for base in basis_row:
            content += f"{base} "
        content += "\n"
    element_list_map = contcar.get_element_list_map()

    element_str = ""
    count_str = ""
    for element, element_list in element_list_map.items():
        if element == "X":
            continue
        element_str += element + " "
        count_str += str(len(element_list)) + " "
    content += element_str + "\n" + count_str + "\n"
    content += "selective\n"
    content += "Direct\n"
    for element, element_list in element_list_map.items():
        if element == "X":
            continue
        for index in element_list:
            if index in atom_index_list:
                content += np.array2string(contcar.atom_list[int(index)].fractional_position,
                                           formatter={"float_kind": lambda x: "%.16f" % x})[1:-1] + " T T T" + "\n"
            else:
                content += np.array2string(contcar.atom_list[int(index)].fractional_position,
                                           formatter={"float_kind": lambda x: "%.16f" % x})[1:-1] + " F F F" + "\n"
    with open(out_filename, "w") as f:
        f.write(content)


def prepare_incar(out_filename) -> None:
    content = """\
NWRITE = 2

PREC   = Accurate
ISYM   = 2
NELM   = 240
NELMIN = 4

NSW    = 10000
IBRION = 5
POTIM  = 0.015
NFREE  = 2
ISIF   = 2

ISMEAR = 1
SIGMA  = 0.4

IALGO  = 48
LREAL  = AUTO
ENCUT  = 450.00
ENAUG  = 600.00
EDIFF  = 1e-6
ISPIN  = 1

LWAVE  = .FALSE.
LCHARG = .FALSE.
"""
    with open(out_filename, "w") as f:
        f.write(content)

# if __name__ == "__main__":
#     config_s = cfg.read_config(
#         f'/Users/zhucongx/qm_srv1/GOALI_DFT_BACKUP/gm/Compiled/MgZnSn_1/240_5_5_5_1/config0/s/start.cfg')
#     config_e = cfg.read_config(
#         f'/Users/zhucongx/qm_srv1/GOALI_DFT_BACKUP/gm/Compiled/MgZnSn_1/240_5_5_5_1/config0/e_0/end.cfg')
#     v_id, j_id = cfg.find_jump_pair_from_cfg(config_s, config_e)
#     print(v_id, j_id)
#     prepare_zpe_poscar(
#         '/Users/zhucongx/qm_srv1/GOALI_DFT_BACKUP/gm/Compiled/MgZnSn_1/240_5_5_5_1/config0/NEB_0/00/POSCAR',
#         '/Users/zhucongx/qm_srv1/GOALI_DFT_BACKUP/gm/Compiled/MgZnSn_1/240_5_5_5_1/config0/s/POSCAR',
#         '/Users/zhucongx/qm_srv1/GOALI_DFT_BACKUP/gm/Compiled/MgZnSn_1/240_5_5_5_1/config0/s/start.cfg',
#         'TEST', j_id, )
