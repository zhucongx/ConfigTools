import numpy as np
import typing
from LMClib.cfg.config import *


def generate_fcc100(element: Element,
                    factor: typing.Tuple[int, int, int],
                    lattice_constant: float) -> Config:
    nx, ny, nz = factor
    basis = lattice_constant * np.array(factor) * np.eye(3, dtype=np.float64)
    element_vector: typing.List[Element] = list()
    positions = []
    unitcell_positions = [(x, y, z) for x in range(nx) for y in range(ny) for z in range(nz)]
    site_positions = [(0, 0, 0), (1 / 2, 1 / 2, 0), (1 / 2, 0, 1 / 2), (0, 1 / 2, 1 / 2)]

    for x, y, z in unitcell_positions:
        for dx, dy, dz in site_positions:
            element_vector.append(element)
            positions.append(np.array([(x + dx) / nx, (y + dy) / ny, (z + dz) / nz]))

    return Config(basis, np.array(positions).T, element_vector)


def generate_fcc111(element: Element,
                    factor: typing.Tuple[int, int, int],
                    lattice_constant: float) -> Config:
    nx, ny, nz = factor
    basis = np.array(
        (lattice_constant * np.sqrt(2) / 2 * nx, lattice_constant * np.sqrt(6) / 2 * ny,
         lattice_constant * np.sqrt(3) * nz)) * np.eye(3, dtype=np.float64)
    element_vector: typing.List[Element] = list()
    positions = []
    unitcell_positions = [(x, y, z) for x in range(nx) for y in range(ny) for z in range(nz)]
    site_positions = [(0, 0, 0), (0, 2 / 3, 2 / 3), (0, 1 / 3, 1 / 3),
                      (1 / 2, 5 / 6, 1 / 3), (1 / 2, 1 / 6, 2 / 3), (1 / 2, 1 / 2, 0)]

    for x, y, z in unitcell_positions:
        for dx, dy, dz in site_positions:
            element_vector.append(element)
            positions.append(
                np.array([(x + dx) / nx, (y + dy) / ny, (z + dz) / nz]))

    return Config(basis, np.array(positions).T, element_vector)


def integrate_fcc111_from_fcc100(config111, config100):
    basis_111 = config111.get_basis()
    basis_100 = config100.get_basis()

    cartesian_position_matrix_111 = config111.get_cartesian_positions_matrix()
    relative_position_matrix_111 = config111.get_relative_positions_matrix()

    cartesian_position_matrix_100 = config100.get_cartesian_positions_matrix()
    relative_position_matrix_100 = config100.get_relative_positions_matrix()

    rotation_matrix = np.array([[np.sqrt(2) / 2, -np.sqrt(2) / 2, 0],
                                [np.sqrt(6) / 6, np.sqrt(6) / 6, -np.sqrt(6) / 3],
                                [np.sqrt(3) / 3, np.sqrt(3) / 3, np.sqrt(3) / 3]],
                               dtype=np.float64)

    config100.set_basis(rotation_matrix.dot(basis_100.T).T)
    config100.set_cartesian_positions_matrix(rotation_matrix.dot(cartesian_position_matrix_100))
    config100.move_cartesian((EPSILON, EPSILON, EPSILON))
    config111.append(config100)
    return config111


if __name__ == '__main__':
    import ase.io

    # from ase.build import fcc111
    # slab = fcc111('Al', size=(20, 20, 24), a=4.046, orthogonal=True, periodic=True)
    # ase.io.write("test_out.cfg", slab, format="cfg")

    cfg111 = generate_fcc111(Element.Al, (20, 10, 8), 4.046)

    cfg100 = generate_fcc100(Element.Mg, (4, 4, 4), 4.046)

    cfg = integrate_fcc111_from_fcc100(cfg111, cfg100)

    ase.io.write("test_out.xyz", Config.to_ase(cfg), format="extxyz")

    # rotation_matrix = np.array([[1, -1, 0], [1, 1, -2], [1, 1, 1]], dtype=np.float64)
    # rotation_matrix = 1 / 6 * np.array([[3 * np.sqrt(2), np.sqrt(6), 2 * np.sqrt(3)],
    #                                     [-3 * np.sqrt(2), np.sqrt(6), 2 * np.sqrt(3)],
    #                                     [0, -2 * np.sqrt(6), 2 * np.sqrt(3)]], dtype=np.float64)
