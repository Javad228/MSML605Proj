import torch
import numpy as np
from torchvision import datasets, transforms
import time
import os
import multiprocessing

os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(multiprocessing.cpu_count())
torch.set_num_threads(multiprocessing.cpu_count())

def tensorfile(fp, shp):
    dat = np.fromfile(fp, dtype=np.float32)
    return torch.from_numpy(dat.reshape(shp))

class VariantNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mod1 = torch.nn.Conv2d(1, 8, 3, padding=1)
        self.mod2 = torch.nn.Linear(14*14*8, 10)
    def forward(self, z):
        if z.dtype != torch.float32:
            z = z.float()
        if z.dim() == 3:
            z = z.unsqueeze(0)
        y = torch.relu(self.mod1(z))
        y = torch.max_pool2d(y, 2)
        y = self.mod2(y.view(y.size(0), -1))
        return y

def run_cpu():
    cpus = multiprocessing.cpu_count()
    mdl = VariantNet()
    t_init = time.time()
    w_a = tensorfile('conv1.w', (8, 3, 3, 1)).permute(0, 3, 1, 2).contiguous()
    b_a = tensorfile('conv1.b', (8,))
    w_b = tensorfile('fc.w', (10, 1568))
    b_b = tensorfile('fc.b', (10,))
    idxs = [(c*14*14 + h*14 + w) for h in range(14) for w in range(14) for c in range(8)]
    rev = np.argsort(idxs)
    w_b2 = w_b[:, rev]
    with torch.no_grad():
        mdl.mod1.weight.copy_(w_a)
        mdl.mod1.bias.copy_(b_a)
        mdl.mod2.weight.copy_(w_b2)
        mdl.mod2.bias.copy_(b_b)
    scripted_net = torch.jit.script(mdl)
    scripted_net.eval()
    tfm = transforms.Compose([transforms.ToTensor()])
    batch_sizes = list(range(1000, 10001, 1000))
    for N in batch_sizes:
        loader = torch.utils.data.DataLoader(
            datasets.MNIST('.', train=False, download=True, transform=tfm),
            batch_size=N, shuffle=False, pin_memory=True,
            num_workers=min(4, cpus), persistent_workers=True)
        total = match = 0
        t1 = time.time()
        for u, _ in torch.utils.data.DataLoader(
                datasets.MNIST('.', train=False, transform=tfm), batch_size=10):
            with torch.no_grad():
                scripted_net(u)
            break
        with torch.no_grad():
            for u, v in loader:
                pred = scripted_net(u).argmax(1)
                match += (pred == v).sum().item()
                total += v.size(0)
        t2 = time.time()
        acc = 100*match/total
        thr = total/(t2-t1)
        print(f"{t2-t1:.2f} {thr:.1f}")
        print(f"CPU inference accuracy: {acc:.2f}% throughput: {thr:.1f} img/s")

if __name__ == "__main__":
    run_cpu()
