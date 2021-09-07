from ansys.vac_jump import get_symmetrically_sorted_atom_vectors
import torch
from torch.nn import Linear, Sequential
import torch.nn.functional as F
from torch_geometric.nn import CGConv, NNConv, GCNConv
from torch_geometric.nn import global_mean_pool


class Net1(torch.nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = CGConv(channels=3, dim=9)
        # self.conv2 = CGConv(channels=3, dim=6)
        self.lin = Linear(177, 1)

    def forward(self, sample):
        x, edge_index, edge_attr = sample.x, sample.edge_index, sample.edge_attr
        batch = sample.batch
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        # x = self.conv2(x, edge_index, edge_attr)
        # x = F.relu(x)

        # # 2. Readout layer
        # print(x.shape)

        # x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        #
        # # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        x = torch.flatten(x, )
        x = self.lin(x)

        return x

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Net2(torch.nn.Module):
    pass


#     def __init__(self):
#         super(Net2, self).__init__()
#         self.conv1 = GCNConv(3, 12)
#         self.conv2 = GCNConv(12, 12)
#         self.lin = Linear(12, 1)
#
#     def forward(self, sample):
#         x, edge_index, edge_attr = sample.x, sample.edge_index, sample.edge_attr
#         batch = sample.batch
#         # 1. Obtain node embeddings
#         x = self.conv1(x, edge_index, edge_attr)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index, edge_attr)
#
#         x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin(x)
#         return x
#
#     def count_parameters(model):
#         return sum(p.numel() for p in model.parameters() if p.requires_grad)
#

if __name__ == "__main__":
    model1 = Net1()
    print(model1)
    print(model1.count_parameters())

    # model2 = Net2()
    # print(model2)
    # print(model2.count_parameters())
