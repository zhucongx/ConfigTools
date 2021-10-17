import os
from cfg.config import *
from ansys.vac_jump import _get_symmetrically_sorted_atom_vectors
import typing
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_networkx
from tqdm import tqdm

atom_encoding_dict = {"Al": [0, 0, 1], "Mg": [0, 1, 0], "Zn": [1, 0, 0]}
bond_encoding_dict = {("Al", "Al"): [0, 0, 0, 0, 0, 0, 0, 0, 1],
                      ("Mg", "Mg"): [0, 0, 0, 0, 0, 0, 0, 1, 0],
                      ("Zn", "Zn"): [0, 0, 0, 0, 0, 0, 1, 0, 0],
                      ("Al", "Mg"): [0, 0, 0, 0, 0, 1, 0, 0, 0],
                      ("Zn", "Mg"): [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      ("Al", "Zn"): [0, 0, 0, 1, 0, 0, 0, 0, 0],
                      ("Mg", "Al"): [0, 0, 1, 0, 0, 0, 0, 0, 0],
                      ("Mg", "Zn"): [0, 1, 0, 0, 0, 0, 0, 0, 0],
                      ("Zn", "Al"): [1, 0, 0, 0, 0, 0, 0, 0, 0]}


def build_data_from_config(config: Config,
                           jump_pair: typing.Tuple[int, int],
                           barrier: typing.Tuple[float, float]) -> typing.List[Data]:
    atom_vectors = _get_symmetrically_sorted_atom_vectors(config, jump_pair)
    data_list = []
    i = 0
    for atom_vector in atom_vectors:
        edge_index = []
        x = []
        edge_attr = []

        for atom in atom_vector:
            if atom.elem_type == "X":
                continue
            x.append(atom_encoding_dict[atom.elem_type])

            for index in atom.first_nearest_neighbor_list:
                if atom_vector[index].elem_type == "X":
                    continue
                edge_index.append([atom.atom_id, index])
                edge_attr.append(bond_encoding_dict[(atom.elem_type, atom_vector[index].elem_type)])

        data_list.append(Data(x=torch.tensor(x, dtype=torch.float),
                              edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                              y=torch.tensor([barrier[i]], dtype=torch.float),
                              edge_attr=torch.tensor(edge_attr, dtype=torch.float)))
        i += 1
    return data_list


class ConfigDataset(InMemoryDataset):
    def __init__(self, root="./data", transform=None, pre_transform=None):
        super(ConfigDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["some_file_1", "some_file_2", ...]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        with open(os.path.join(self.raw_dir, "barriers.txt"), "r") as f:
            lines = f.readlines()
        for i in tqdm(range(len(lines)), desc="Loading configs ..."):
            # for i in tqdm(range(5), desc="Loading configs ..."):
            dir_path = os.path.join(self.raw_dir, "config" + str(i))
            line_list = lines[i].split()

            config = read_config(os.path.join(dir_path, "start.cfg"))
            jump_pair = (int(line_list[0]), int(line_list[1]))
            barriers = (float(line_list[2]), float(line_list[3]))

            data_tuple = build_data_from_config(config, jump_pair, barriers)
            for data in data_tuple:
                data_list.append(data)

            logging.debug(f"Processing config{i}")
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def visualize(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set1")
        if epoch is not None and loss is not None:
            plt.xlabel(f"Epoch: {epoch}, Loss: {loss.item():.4f}", fontsize=16)
    else:
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                         node_color=color, cmap="Set1")
    plt.show()


if __name__ == "__main__":
    import random

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:: %(message)s")

    dataset = ConfigDataset(root="../data")

    print()
    print(f"Dataset: {dataset}:")
    print("=" * 60)
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")

    dat: Data = dataset[random.randint(0, len(dataset))]
    G = to_networkx(dat, to_undirected=True)
    visualize(G, color=dat.x)

    print()
    print(dat)
    print("=" * 60)
    # Gather some statistics about the first graph.
    print(f"Number of nodes: {dat.num_nodes}")
    print(f"Number of edges: {dat.num_edges}")
    print(f"Average node degree: {dat.num_edges / dat.num_nodes:.2f}")
    print(f"Contains isolated nodes: {dat.contains_isolated_nodes()}")
    print(f"Contains self-loops: {dat.contains_self_loops()}")
    print(f"Is directed: {dat.is_directed()}")
