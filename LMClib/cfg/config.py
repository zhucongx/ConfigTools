import numpy as np
import typing
import sys
import ase
from LMClib.cfg.element import Element

EPSILON = sys.float_info.epsilon * 10


class Config:

    def __init__(self,
                 basis: typing.Optional[np.ndarray] = None,
                 relative_position_matrix: typing.Optional[np.ndarray] = None,
                 element_vector: typing.Optional[typing.List[Element]] = None):
        if basis is None or relative_position_matrix is None or element_vector is None:
            return

        if len(element_vector) != relative_position_matrix.shape[1]:
            raise ValueError(
                "Lattice vector and atom vector size do not match, lattice_vector.size() = " +
                str(relative_position_matrix.shape[1]) + ", element_vector_.size() = " +
                str(len(element_vector)))

        # [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
        self._basis = basis
        self._relative_position_matrix = relative_position_matrix
        self._cartesian_position_matrix = np.dot(basis, relative_position_matrix)
        self._element_vector = element_vector

        self._lattice_to_atom_hashmap = {index: index for index, atom in enumerate(element_vector)}
        self._atom_to_lattice_hashmap = {v: k for k, v in self._lattice_to_atom_hashmap.items()}

        self._periodic_boundary_condition = [True, True, True]
        self._neighbor_lists: typing.List[typing.List[typing.List[int]]] = [[[]]]
        self._cutoffs: typing.List[float] = []

    def get_num_atoms(self) -> int:
        return len(self._element_vector)

    def get_basis(self) -> np.ndarray:
        return self._basis

    def set_basis(self, basis: np.ndarray) -> None:
        self._basis = basis

    def get_neighbor_lists(self) -> typing.List[typing.List[typing.List[int]]]:
        return self._neighbor_lists

    def set_relative_positions_matrix(self, positions: np.ndarray) -> None:
        self._relative_position_matrix = positions
        self._cartesian_position_matrix = np.dot(self._basis, positions)

    def set_cartesian_positions_matrix(self, positions: np.ndarray) -> None:
        self._cartesian_position_matrix = positions
        self._relative_position_matrix = np.linalg.solve(self._basis, positions)

    def get_relative_positions_matrix(self) -> np.ndarray:
        return self._relative_position_matrix

    def get_cartesian_positions_matrix(self) -> np.ndarray:
        return self._cartesian_position_matrix

    def set_periodic_boundary_condition(self,
                                        periodic_boundary_condition: typing.List[bool]) -> None:
        self._periodic_boundary_condition = periodic_boundary_condition

    def move_cartesian(self, displacement_vector: np.ndarray) -> None:
        for col in range(self._cartesian_position_matrix.shape[1]):
            self._cartesian_position_matrix[:, col] += displacement_vector
        self._relative_position_matrix = np.linalg.solve(self._basis,
                                                         self._cartesian_position_matrix)

    def wrap(self) -> None:
        for lattice_id in range(self.get_num_atoms()):
            for kDim, periodic in enumerate(self._periodic_boundary_condition):
                if periodic:
                    # while self._relative_position_matrix[kDim, lattice_id] > 1:
                    #     self._relative_position_matrix[kDim, lattice_id] -= 1
                    # while self._relative_position_matrix[kDim, lattice_id] < 0:
                    #     self._relative_position_matrix[kDim, lattice_id] += 1
                    floor = np.floor(self._relative_position_matrix[kDim, lattice_id])
                    self._relative_position_matrix[kDim, lattice_id] -= floor

        self._cartesian_position_matrix = np.dot(self._basis, self._relative_position_matrix)

    def atom_jump(self, atom_id_jump_pair: typing.Tuple[int, int]) -> None:
        atom_id_lhs, atom_id_rhs = atom_id_jump_pair
        lattice_id_lhs = self._atom_to_lattice_hashmap[atom_id_lhs]
        lattice_id_rhs = self._atom_to_lattice_hashmap[atom_id_rhs]

        self._atom_to_lattice_hashmap[atom_id_lhs] = lattice_id_rhs
        self._atom_to_lattice_hashmap[atom_id_rhs] = lattice_id_lhs
        self._lattice_to_atom_hashmap[lattice_id_lhs] = atom_id_rhs
        self._lattice_to_atom_hashmap[lattice_id_rhs] = atom_id_lhs

    def get_relative_distance_vector_lattice(self, lattice_id1: int,
                                             lattice_id2: int) -> np.ndarray:
        relative_distance_vector = self._relative_position_matrix[:, lattice_id2] - \
                                   self._relative_position_matrix[:, lattice_id1]
        # periodic boundary conditions
        for kDim, periodic in enumerate(self._periodic_boundary_condition):
            if periodic:
                while relative_distance_vector[kDim] >= 0.5:
                    relative_distance_vector[kDim] -= 1
                while relative_distance_vector[kDim] < -0.5:
                    relative_distance_vector[kDim] += 1
        return relative_distance_vector

    def delete_atom(self, atom_id_list: typing.List[int]) -> None:
        for atom_id in atom_id_list:
            lattice_id = self._atom_to_lattice_hashmap[atom_id]
            del self._lattice_to_atom_hashmap[lattice_id]
            del self._atom_to_lattice_hashmap[atom_id]

        self._element_vector = [item for index, item in enumerate(self._element_vector) if
                                index not in atom_id_list]

        self._relative_position_matrix = np.delete(self._relative_position_matrix,
                                                   atom_id_list, axis=1)
        self._cartesian_position_matrix = np.delete(self._cartesian_position_matrix,
                                                    atom_id_list, axis=1)

    def _build_cell(self, cutoff):
        basis_inverse_tran = np.linalg.inv(self._basis).T
        cutoff_max = np.inf
        for kDim, periodic in enumerate(self._periodic_boundary_condition):
            if periodic:
                cutoff_max = min(cutoff_max, 0.5 / np.linalg.norm(basis_inverse_tran[:, kDim]))

        if cutoff_max <= cutoff:
            raise RuntimeError(
                f"The cutoff is larger than the maximum cutoff allowed due to periodic boundary "
                f"conditions, cutoff_max = {cutoff_max}, cutoff_input = {cutoff}")

        self._num_cells = np.floor(np.linalg.norm(self._basis, axis=0) / cutoff).astype(int)
        self._cells: typing.List[typing.List[int]] = [[] for _ in range(np.prod(self._num_cells))]
        for lattice_id in range(self.get_num_atoms()):
            relative_position = self._relative_position_matrix[:, lattice_id]
            cell_pos = np.floor(self._num_cells * relative_position).astype(int)
            cell_idx = \
                (cell_pos[0] * self._num_cells[1] + cell_pos[1]) * self._num_cells[2] + cell_pos[2]
            self._cells[cell_idx].append(lattice_id)

    def update_neighbor_list(self, cutoffs):
        self._cutoffs = sorted(cutoffs)
        self._build_cell(cutoffs[-1])
        self._neighbor_lists: typing.List[typing.List[typing.List[int]]] = \
            [[[] for _ in range(self.get_num_atoms())] for _ in range(len(self._cutoffs))]

        cutoffs_squared = np.square(self._cutoffs)
        offset_list = [(x, y, z) for x in [-1, 0, 1] for y in [-1, 0, 1] for z in [-1, 0, 1]]

        for cell_idx, cell in enumerate(self._cells):
            i = cell_idx // (self._num_cells[1] * self._num_cells[2])
            j = (cell_idx % (self._num_cells[1] * self._num_cells[2])) // self._num_cells[2]
            k = cell_idx % self._num_cells[2]
            for di, dj, dk in offset_list:
                ni = (i + di + self._num_cells[0]) % self._num_cells[0]
                nj = (j + dj + self._num_cells[1]) % self._num_cells[1]
                nk = (k + dk + self._num_cells[2]) % self._num_cells[2]
                neighbor_cell_idx = (ni * self._num_cells[1] + nj) * self._num_cells[2] + nk
                for lattice_id1 in cell:
                    for lattice_id2 in self._cells[neighbor_cell_idx]:
                        if lattice_id2 >= lattice_id1:
                            continue
                        cartesian_distance_squared = np.square(
                            np.dot(self._basis, self.get_relative_distance_vector_lattice(
                                lattice_id1, lattice_id2))).sum()

                        cutoff_id = np.searchsorted(cutoffs_squared,
                                                    cartesian_distance_squared,
                                                    side='left')
                        if cutoff_id < len(cutoffs_squared):
                            self._neighbor_lists[cutoff_id][lattice_id1].append(lattice_id2)
                            self._neighbor_lists[cutoff_id][lattice_id2].append(lattice_id1)

    def append(self, new_config: 'Config', overlap_thresh=1) -> None:
        old_num_atoms = self.get_num_atoms()
        self._cartesian_position_matrix = np.hstack(
            (self._cartesian_position_matrix, new_config.get_cartesian_positions_matrix()))
        self._element_vector.extend(new_config._element_vector)
        self._relative_position_matrix = np.linalg.solve(self._basis,
                                                         self._cartesian_position_matrix)
        for new_id in range(old_num_atoms, self.get_num_atoms()):
            self._lattice_to_atom_hashmap[new_id] = new_id
            self._atom_to_lattice_hashmap[new_id] = new_id
        self.wrap()
        if overlap_thresh == 0:
            return

        self._build_cell(overlap_thresh)
        overlap_thresh_squared = overlap_thresh ** 2
        # The dict is used to store the overlap pairs, the key is the lattice id of the atom in
        # the new config, the value is the lattice id of the atom in the old config
        overlap_dict: typing.Dict[int, int] = {}

        offset_list = [(x, y, z) for x in [-1, 0, 1] for y in [-1, 0, 1] for z in [-1, 0, 1]]
        for cell_idx, cell in enumerate(self._cells):
            i = cell_idx // (self._num_cells[1] * self._num_cells[2])
            j = (cell_idx % (self._num_cells[1] * self._num_cells[2])) // self._num_cells[2]
            k = cell_idx % self._num_cells[2]
            for lattice_id1 in cell:
                if lattice_id1 < old_num_atoms:
                    continue
                for di, dj, dk in offset_list:
                    ni = (i + di + self._num_cells[0]) % self._num_cells[0]
                    nj = (j + dj + self._num_cells[1]) % self._num_cells[1]
                    nk = (k + dk + self._num_cells[2]) % self._num_cells[2]
                    neighbor_cell_idx = (ni * self._num_cells[1] + nj) * self._num_cells[2] + nk
                    for lattice_id2 in self._cells[neighbor_cell_idx]:
                        if lattice_id2 >= lattice_id1:
                            continue
                        cartesian_distance_squared = np.square(
                            np.dot(self._basis, self.get_relative_distance_vector_lattice(
                                lattice_id1, lattice_id2))).sum()
                        if cartesian_distance_squared < overlap_thresh_squared:
                            overlap_dict[lattice_id1] = lattice_id2
        # To remove the overlap, we need to move the atoms in the old config
        self.delete_atom([old_id for new_id, old_id in overlap_dict.items()])
    @staticmethod
    def from_ase(ase_atoms: ase.Atoms) -> 'Config':
        basis = ase_atoms.get_cell()
        relative_position_matrix = ase_atoms.get_scaled_positions().T
        element_vector = [Element.from_symbol(atom.symbol) for atom in ase_atoms]
        return Config(basis, relative_position_matrix, element_vector)

    @staticmethod
    def to_ase(config: 'Config') -> ase.Atoms:
        cell = config.get_basis()
        positions = config._cartesian_position_matrix.T
        symbols = [element.symbol for element in config._element_vector]
        return ase.Atoms(symbols=symbols, positions=positions, cell=cell)


if __name__ == '__main__':
    import ase.io
    from time import perf_counter

    start = perf_counter()
    atoms = ase.io.read("/Users/zhucongx/Research/"
                        "GOALI/LatticeMonteCarlo/test/test_small.cfg", format="cfg")
    cfg = Config.from_ase(atoms)
    # ase.io.write("test_out.cfg", Config.to_ase(config), format="cfg")

    # cfg.update_neighbor_list([3.5, 4.8, 5.3, 5.9, 6.5, 7.1, 7.6, 8.2])
    cfg.update_neighbor_list([3.5])
    end = perf_counter()
    print("Done, time = ", end - start)
