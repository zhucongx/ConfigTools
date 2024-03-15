import numpy as np
import typing
from LMClib.cfg.config import *


def _generate(element_list: typing.List[Element],
              factor: typing.Tuple[int, int, int],
              site_positions_list: typing.List[typing.List[typing.Tuple[float, float, float]]]):
    nx, ny, nz = factor
    element_vector: typing.List[Element] = list()
    positions = []
    unitcell_positions = [(x, y, z) for x in range(nx) for y in range(ny) for z in range(nz)]
    for x, y, z in unitcell_positions:
        for element, site_positions in zip(element_list, site_positions_list):
            for dx, dy, dz in site_positions:
                element_vector.append(element)
                positions.append(
                    np.array([(x + dx) / nx, (y + dy) / ny, (z + dz) / nz]))

    return np.array(positions).T, element_vector


def generate_fcc001(element: Element,
                    factor: typing.Tuple[int, int, int],
                    lattice_constant: float) -> Config:
    # oriented X=[100] Y=[010] Z=[001].
    basis = lattice_constant * np.array(factor) * np.eye(3, dtype=np.float64)
    site_positions = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    return Config(basis,
                  *_generate([element], factor, [site_positions]))


def generate_fcc111(element: Element,
                    factor: typing.Tuple[int, int, int],
                    lattice_constant: float) -> Config:
    # oriented X=[1-10] Y=[11-2] Z=[111].
    basis = lattice_constant * np.array(factor) * np.diag((np.sqrt(2) / 2,
                                                           np.sqrt(6) / 2,
                                                           np.sqrt(3)))

    site_positions = [(0, 0, 0), (0, 2 / 3, 2 / 3), (0, 1 / 3, 1 / 3),
                      (0.5, 5 / 6, 1 / 3), (0.5, 1 / 6, 2 / 3), (0.5, 0.5, 0)]
    return Config(basis,
                  *_generate([element], factor, [site_positions]))


def generate_bcc001(element: Element,
                    factor: typing.Tuple[int, int, int],
                    lattice_constant: float) -> Config:
    # oriented X=[100] Y=[010] Z=[001].
    basis = lattice_constant * np.array(factor) * np.eye(3, dtype=np.float64)
    site_positions = [(0, 0, 0), (0.5, 0.5, 0.5)]
    return Config(basis,
                  *_generate([element], factor, [site_positions]))


def generate_bcc111(element: Element,
                    factor: typing.Tuple[int, int, int],
                    lattice_constant: float) -> Config:
    # oriented X=[1-10] Y=[11-2] Z=[111].
    basis = lattice_constant * np.array(factor) * np.diag((np.sqrt(2),
                                                           np.sqrt(6),
                                                           np.sqrt(3) / 2))
    site_positions = [(0, 0, 0), (0, 2 / 3, 2 / 3), (0, 1 / 3, 1 / 3),
                      (0.5, 5 / 6, 1 / 3), (0.5, 1 / 6, 2 / 3), (0.5, 0.5, 0)]
    return Config(basis,
                  *_generate([element], factor, [site_positions]))


def generate_l10(element1: Element, element2: Element,
                 factor: typing.Tuple[int, int, int],
                 lattice_constant: float) -> Config:
    basis = lattice_constant * np.array(factor) * np.eye(3, dtype=np.float64)
    site_positions1 = [(0, 0, 0), (0.5, 0.5, 0)]
    site_positions2 = [(0.5, 0, 0.5), (0, 0.5, 0.5)]
    return Config(basis,
                  *_generate([element1, element2], factor, [site_positions1, site_positions2]))


def generate_l11(element1: Element, element2: Element,
                 factor: typing.Tuple[int, int, int],
                 lattice_constant: float) -> Config:
    basis = lattice_constant * np.array(factor) * np.diag((2, 2, 2))
    site_positions1 = [(0.25, 0.25, 0.00), (0.25, 0.00, 0.25), (0.00, 0.25, 0.25),
                       (0.00, 0.00, 0.50), (0.00, 0.50, 0.00), (0.25, 0.75, 0.50),
                       (0.25, 0.50, 0.75), (0.00, 0.75, 0.75), (0.50, 0.00, 0.00),
                       (0.75, 0.25, 0.50), (0.75, 0.00, 0.75), (0.50, 0.25, 0.75),
                       (0.75, 0.75, 0.00), (0.75, 0.50, 0.25), (0.50, 0.75, 0.25),
                       (0.50, 0.50, 0.50)]
    site_positions2 = [(0.00, 0.00, 0.00), (0.25, 0.25, 0.50), (0.25, 0.00, 0.75),
                       (0.00, 0.25, 0.75), (0.25, 0.75, 0.00), (0.25, 0.50, 0.25),
                       (0.00, 0.75, 0.25), (0.00, 0.50, 0.50), (0.75, 0.25, 0.00),
                       (0.75, 0.00, 0.25), (0.50, 0.25, 0.25), (0.50, 0.00, 0.50),
                       (0.50, 0.50, 0.00), (0.75, 0.75, 0.50), (0.75, 0.50, 0.75),
                       (0.50, 0.75, 0.75)]

    return Config(basis,
                  *_generate([element1, element2], factor, [site_positions1, site_positions2]))


def generate_l12(element1: Element, element2: Element,
                 factor: typing.Tuple[int, int, int],
                 lattice_constant: float) -> Config:
    basis = lattice_constant * np.array(factor) * np.eye(3, dtype=np.float64)
    site_positions1 = [(0, 0, 0)]
    site_positions2 = [(0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    return Config(basis,
                  *_generate([element1, element2], factor, [site_positions1, site_positions2]))


def generate_l10star(element1: Element, element2: Element,
                     factor: typing.Tuple[int, int, int],
                     lattice_constant: float) -> Config:
    basis = lattice_constant * np.array(factor) * np.diag((2, 1, 1))
    site_positions1 = [(0, 0, 0), (0.5, 0, 0), (0.75, 0.5, 0), (0.25, 0, 0.5)]
    site_positions2 = [(0.25, 0.5, 0), (0.75, 0, 0.5), (0, 0.5, 0.5), (0.5, 0.5, 0.5)]
    return Config(basis,
                  *_generate([element1, element2], factor, [site_positions1, site_positions2]))


def generate_l12star(element1: Element, element2: Element,
                     factor: typing.Tuple[int, int, int],
                     lattice_constant: float) -> Config:
    basis = lattice_constant * np.array(factor) * np.diag((4, 1, 1))
    site_positions1 = [(0.5, 0.5, 0.5), (0.0, 0.0, 0.0), (0.75, 0.0, 0.0), (0.25, 0.5, 0.5)]
    site_positions2 = [(0.125, 0.5, 0.0), (0.25, 0.0, 0.0), (0.375, 0.5, 0.0), (0.5, 0.0, 0.0),
                       (0.625, 0.5, 0.0), (0.875, 0.5, 0.0), (0.0, 0.5, 0.5), (0.125, 0.0, 0.5),
                       (0.375, 0.0, 0.5), (0.625, 0.0, 0.5), (0.75, 0.5, 0.5), (0.875, 0.0, 0.5)]

    return Config(basis,
                  *_generate([element1, element2], factor, [site_positions1, site_positions2]))


def generate_z1(element1: Element, element2: Element,
                factor: typing.Tuple[int, int, int],
                lattice_constant: float) -> Config:
    basis = lattice_constant * np.array(factor) * np.diag((1, 1, 2))
    site_positions1 = [(0.5, 0.5, 0.0), (0.0, 0.0, 0.0)]
    site_positions2 = [(0.0, 0.5, 0.25), (0.5, 0.0, 0.25), (0.0, 0.0, 0.5),
                       (0.5, 0.5, 0.5), (0.0, 0.5, 0.75), (0.5, 0.0, 0.75)]
    return Config(basis,
                  *_generate([element1, element2], factor, [site_positions1, site_positions2]))


def generate_b2(element1: Element, element2: Element,
                factor: typing.Tuple[int, int, int],
                lattice_constant: float) -> Config:
    basis = lattice_constant * np.array(factor) * np.eye(3, dtype=np.float64)
    site_positions1 = [(0.00, 0.00, 0.00)]
    site_positions2 = [(0.50, 0.50, 0.50)]
    return Config(basis,
                  *_generate([element1, element2], factor,
                             [site_positions1, site_positions2]))


def generate_l21(element1: Element, element2: Element, element3: Element,
                 factor: typing.Tuple[int, int, int],
                 lattice_constant: float) -> Config:
    basis = lattice_constant * np.array(factor) * np.diag((2, 2, 2))
    site_positions1 = [(0.00, 0.00, 0.50), (0.00, 0.50, 0.00), (0.50, 0.00, 0.00),
                       (0.50, 0.50, 0.50)]
    site_positions2 = [(0.00, 0.00, 0.00), (0.00, 0.50, 0.50), (0.50, 0.00, 0.50),
                       (0.50, 0.50, 0.00)]
    site_positions3 = [(0.25, 0.25, 0.75), (0.25, 0.75, 0.75), (0.25, 0.75, 0.25),
                       (0.25, 0.25, 0.25), (0.75, 0.25, 0.25), (0.75, 0.75, 0.25),
                       (0.75, 0.75, 0.75), (0.75, 0.25, 0.75)]
    return Config(basis,
                  *_generate([element1, element2, element3], factor,
                             [site_positions1, site_positions2, site_positions3]))


def generate_l22(element1: Element, element2: Element,
                 factor: typing.Tuple[int, int, int],
                 lattice_constant: float) -> Config:
    basis = lattice_constant * np.array(factor) * np.diag((3, 3, 3))
    site_positions1 = [(0, 2 / 3, 0), (0, 1 / 3, 0), (1 / 3, 0, 0),
                       (0, 0, 2 / 3), (2 / 3, 0, 0), (0, 0, 1 / 3),
                       (1 / 2, 1 / 6, 1 / 2), (1 / 2, 5 / 6, 1 / 2), (5 / 6, 1 / 2, 1 / 2),
                       (1 / 2, 1 / 2, 1 / 6), (1 / 6, 1 / 2, 1 / 2), (1 / 2, 1 / 2, 5 / 6)]
    site_positions2 = [(1 / 6, 1 / 2, 1 / 6), (5 / 6, 1 / 2, 1 / 6), (5 / 6, 5 / 6, 1 / 6),
                       (1 / 3, 0, 1 / 3), (1 / 3, 1 / 3, 1 / 3), (5 / 6, 5 / 6, 1 / 2),
                       (1 / 6, 5 / 6, 5 / 6), (1 / 2, 5 / 6, 5 / 6), (1 / 3, 2 / 3, 2 / 3),
                       (1 / 2, 1 / 6, 1 / 6), (1 / 6, 1 / 6, 1 / 6), (1 / 6, 1 / 6, 1 / 2),
                       (1 / 3, 2 / 3, 0), (1 / 3, 2 / 3, 1 / 3), (1 / 6, 5 / 6, 1 / 6),
                       (0, 0, 0), (1 / 6, 5 / 6, 1 / 2), (1 / 6, 1 / 2, 5 / 6),
                       (0, 2 / 3, 1 / 3), (1 / 2, 5 / 6, 1 / 6), (2 / 3, 2 / 3, 1 / 3),
                       (2 / 3, 0, 2 / 3), (1 / 3, 0, 2 / 3), (1 / 3, 1 / 3, 2 / 3),
                       (5 / 6, 1 / 2, 5 / 6), (5 / 6, 5 / 6, 5 / 6), (1 / 3, 1 / 3, 0),
                       (2 / 3, 1 / 3, 1 / 3), (0, 1 / 3, 1 / 3), (5 / 6, 1 / 6, 1 / 6),
                       (0, 2 / 3, 2 / 3), (2 / 3, 2 / 3, 2 / 3), (2 / 3, 2 / 3, 0),
                       (5 / 6, 1 / 6, 1 / 2), (5 / 6, 1 / 6, 5 / 6), (2 / 3, 1 / 3, 2 / 3),
                       (1 / 2, 1 / 2, 1 / 2), (2 / 3, 1 / 3, 0), (2 / 3, 0, 1 / 3),
                       (1 / 2, 1 / 6, 5 / 6), (0, 1 / 3, 2 / 3), (1 / 6, 1 / 6, 5 / 6)]

    return Config(basis,
                  *_generate([element1, element2], factor,
                             [site_positions1, site_positions2]))


def integrate_small_small_to_large(config_large, config_small):
    config_large.append(config_small, 1)
    return config_large


def integrate_fcc111_from_fcc001(config111, config100):
    rotation_matrix = np.array([[np.sqrt(2) / 2, -np.sqrt(2) / 2, 0],
                                [np.sqrt(6) / 6, np.sqrt(6) / 6, -np.sqrt(6) / 3],
                                [np.sqrt(3) / 3, np.sqrt(3) / 3, np.sqrt(3) / 3]],
                               dtype=np.float64)

    # config100.set_basis(rotation_matrix.dot(config100.get_basis().T).T)
    cartesian_position_matrix_100 = config100.get_cartesian_positions_matrix()
    config100.set_cartesian_positions_matrix(rotation_matrix.dot(cartesian_position_matrix_100))
    config100.move_cartesian((EPSILON, EPSILON, EPSILON))

    config111.append(config100, 1)
    return config111


if __name__ == '__main__':
    import ase.io

    l10 = generate_l10(Element.Mg, Element.Zn, (2, 1, 1), 4.046)
    fcc = generate_fcc001(Element.Al, (4, 4, 4), 4.046)
    fcc.append(l10, 1)
    ase.io.write("l12.xyz", Config.to_ase(fcc), format="extxyz")

# from ase.build import fcc111
# slab = fcc111('Al', size=(20, 20, 24), a=4.046, orthogonal=True, periodic=True)
# ase.io.write("test_out.cfg", slab, format="cfg")

# cfg111 = generate_fcc111(Element.Al, (20, 10, 8), 4.046)
# ase.io.write("Al111.xyz", Config.to_ase(cfg111), format="extxyz")

# cfg111 = generate_fcc111(Element.Al, (20, 10, 8), 4.046)
# # cfg100 = generate_fcc100(Element.Mg, (4, 4, 4), 4.046)
# cfg100 = Config.from_ase(ase.io.read("/Users/zhucongx/Research/"
#                                      "GOALI/ConfigTools/test/test_files/forward.cfg",
#                                      format="cfg"))
# cfg = integrate_fcc111_from_fcc100(cfg111, cfg100)
# ase.io.write("test_out1.xyz", Config.to_ase(cfg), format="extxyz")
#
# cfg111 = generate_fcc111(Element.Al, (20, 10, 8), 4.046)
# # cfg100 = generate_fcc100(Element.Mg, (4, 4, 4), 4.046)
# cfg100 = Config.from_ase(ase.io.read("/Users/zhucongx/Research/"
#                                      "GOALI/ConfigTools/test/test_files/backward.cfg",
#                                      format="cfg"))
# cfg = integrate_fcc111_from_fcc100(cfg111, cfg100)
# # cfg.reassign_lattice()
# ase.io.write("test_out2.xyz", Config.to_ase(cfg), format="extxyz")
