from cfg.atom import Atom
import typing


class Cluster(object):
    def __init__(self, *atoms: typing.Union[Atom, typing.List[Atom]]):
        if len(atoms) == 1 and isinstance(atoms[0], list):
            self.__atom_list: typing.List[Atom] = atoms[0]
        else:
            self.__atom_list: typing.List[Atom] = list()
            for insert_atom in atoms:
                self.__atom_list.append(insert_atom)

        self.__atom_list.sort(key=lambda sort_atom: sort_atom.atom_id)

    @property
    def atom_list(self) -> typing.List[Atom]:
        return self.__atom_list
