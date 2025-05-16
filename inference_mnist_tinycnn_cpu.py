import torch
import numpy as np
from torchvision import datasets, transforms
import time

class NetArch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_a = torch.nn.Conv2d(1, 8, 3, padding=1)
        self.layer_b = torch.nn.Linear(14*14*8, 10)
    def forward(self, z):
        z = torch.relu(self.layer_a(z))
        z = torch.max_pool2d(z, 2)
        z = self.layer_b(z.view(z.size(0), -1))
        return z

def tensor_from_file(path, dims):
    data = np.fromfile(path, dtype=np.float32)
    return torch.from_numpy(data.reshape(dims))

model = NetArch()
weights_a = tensor_from_file('conv1.w', (8, 3, 3, 1)).permute(0, 3, 1, 2).contiguous()
biases_a = tensor_from_file('conv1.b', (8,))
weights_b = tensor_from_file('fc.w', (10, 1568))
biases_b = tensor_from_file('fc.b', (10,))
indices = []
for i in range(14):
    for j in range(14):
        for k in range(8):
            indices.append(k*14*14 + i*14 + j)
reverse = np.argsort(indices)
weights_b2 = weights_b[:, reverse]
with torch.no_grad():
    model.layer_a.weight.copy_(weights_a)
    model.layer_a.bias.copy_(biases_a)
    model.layer_b.weight.copy_(weights_b2)
    model.layer_b.bias.copy_(biases_b)
model.eval()

transformer = transforms.Compose([transforms.ToTensor()])
data_iter = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=False, download=True, transform=transformer),
    batch_size=1000, shuffle=False)

hits = count = 0
t0 = time.time()
with torch.no_grad():
    for u, v in data_iter:
        y_hat = model(u).argmax(1)
        hits += (y_hat == v).sum().item()
        count += v.size(0)
t1 = time.time()
acc = 100*hits/count
thr = count/(t1-t0)
print(f"CPU inference accuracy: {acc:.2f}% throughput: {thr:.1f} img/s")
