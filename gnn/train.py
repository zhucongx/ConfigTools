import math
import torch
import logging
import numpy as np
from model import Net1, Net2
from dataset import ConfigDataset
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(levelname)s:: %(message)s')
torch.manual_seed(12345)
dataset = ConfigDataset(root="../data")
dataset = dataset.shuffle()

train_dataset = dataset[:-500]
test_dataset = dataset[-500:]
print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# for step, data in enumerate(train_loader):
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of graphs in the current batch: {data.num_graphs}')
#     print(data)
#     print()

model = Net1()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

plt.figure(figsize=(10, 10))


def train():
    model.train()
    for dat in train_loader:  # Iterate in batches over the training dataset.
        out = model(dat)  # Perform a single forward pass.
        loss = criterion(out, dat.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(loader, filename):
    model.eval()
    sum_mse = 0
    true_y = []
    pred_y = []
    for dat in loader:  # Iterate in batches over the training/test dataset.
        out = model(dat)
        sum_mse += float(((out - dat.y) ** 2).sum())
        true_y.append(dat.y.detach().numpy())
        pred_y.append(out.detach().numpy())

        plt.scatter(dat.y.detach().numpy(), out.detach().numpy(), color="orange")
        plt.xlim([0, 1.2])
        plt.ylim([0, 1.2])
    true_y = np.array(true_y)
    pred_y = np.array(pred_y)
    np.savetxt("checkpoints" + filename + '.true', true_y, fmt='%10.16f')
    np.savetxt("checkpoints" + filename + '.pred', pred_y, fmt='%10.16f')

    plt.title('')
    plt.xlabel('real')
    plt.ylabel('pred')
    return sum_mse / len(loader.dataset)


# view data


for epoch in range(1, 5001):
    plt.clf()
    train()
    train_loss = test(train_loader, 'train')
    test_loss = test(test_loader, 'test')
    if epoch % 10 == 0:
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}')
