import typing
import numpy as np
import configtools.cfg as cfg


def prepare_zpe_poscar(contcar_filename: str,
                       poscar_filename: str,
                       config_filename: str,
                       out_filename: str,
                       jump_atom_index: int) -> None:
    contcar = cfg.read_poscar(contcar_filename, False)
    poscar = cfg.read_poscar(poscar_filename, False)
    config = cfg.read_config(config_filename, False)
    # get jump atom index in poscar which should be same as that in contcar
    jump_atom = config.atom_list[jump_atom_index]
    jump_atom_elem_type = jump_atom.elem_type

    poscar_jump_atom_index = None
    min_relative_distance = 10
    for atom in poscar.atom_list:
        if atom.elem_type != jump_atom_elem_type:
            continue
        relative_distance_vector = cfg.get_relative_distance_vector(atom, jump_atom)
        relative_distance = np.inner(relative_distance_vector, relative_distance_vector)
        if relative_distance < min_relative_distance:
            min_relative_distance = relative_distance
            poscar_jump_atom_index = atom.atom_id

    _write_selective_poscar(contcar, poscar_jump_atom_index, out_filename)


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
EDIFF  = 1e-7
ISPIN  = 1

LWAVE  = .FALSE.
LCHARG = .FALSE.
"""
    with open(out_filename, "w") as f:
        f.write(content)


def _write_selective_poscar(contcar: cfg.Config, jump_atom_index: int, out_filename: str) -> None:
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
            if index == jump_atom_index:
                content += np.array2string(contcar.atom_list[int(index)].relative_position,
                                           formatter={"float_kind": lambda x: "%.16f" % x})[1:-1] + " T T T" + "\n"
            else:
                content += np.array2string(contcar.atom_list[int(index)].relative_position,
                                           formatter={"float_kind": lambda x: "%.16f" % x})[1:-1] + " F F F" + "\n"
    with open(out_filename, "w") as f:
        f.write(content)


# if __name__ == "__main__":
#     prepare_zpe_poscar('/Volumes/LaCie/GOALI_DFT_BACKUP/new/ordered/config0/s/CONTCAR',
#                        '/Volumes/LaCie/GOALI_DFT_BACKUP/new/ordered/config0/s/POSCAR',
#                        '/Volumes/LaCie/GOALI_DFT_BACKUP/new/ordered/config0/s/start.cfg',
#                        'TEST', 83)
