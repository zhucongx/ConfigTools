from cfg.config import Atom
import copy
import typing


class Cluster(object):
    def __init__(self, *atoms: Atom):
        self._atom_list: typing.List[Atom] = list()
        for insert_atom in atoms:
            self._atom_list.append(copy.deepcopy(insert_atom))

        self._atom_list.sort(key=lambda sort_atom: sort_atom.atom_id)

    def __eq__(self, other):
        for atom1, atom2 in zip(self.atom_list, other.atom_list):
            if atom1.atom_id != atom2.atom_id:
                return False
        return True

    def __hash__(self):
        the_hash = hash(self._atom_list[0].atom_id)
        if len(self._atom_list) > 1:
            for atom in self._atom_list[1:]:
                the_hash = the_hash ^ hash(atom.atom_id)
        return the_hash

    @property
    def atom_list(self) -> typing.List[Atom]:
        return self._atom_list

    @property
    def type_key(self) -> str:
        key = ''
        for atom in self._atom_list:
            key += atom.elem_type
        return key
