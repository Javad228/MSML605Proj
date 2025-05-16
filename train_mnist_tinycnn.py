import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

BATCH  = 128
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  8, 3, padding=1)   # 1→8
        self.fc    = nn.Linear(14*14*8, 10)

    def forward(self,x):
        x = torch.relu(self.conv1(x))          # 28×28
        x = torch.max_pool2d(x, 2)             # 14×14
        x = self.fc(x.view(x.size(0), -1))
        return x

net = TinyCNN().to(DEVICE)

tr = transforms.Compose([transforms.ToTensor()])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(".", train=True,  download=True, transform=tr),
    batch_size=BATCH, shuffle=True)
test_loader  = torch.utils.data.DataLoader(
    datasets.MNIST(".", train=False, download=True, transform=tr),
    batch_size=1000, shuffle=False)

opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(1, EPOCHS+1):
    net.train()
    for x,y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        x,y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); loss_fn(net(x), y).backward(); opt.step()

net.eval(); correct = total = 0
with torch.no_grad():
    for x,y in test_loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        preds = net(x).argmax(1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
print(f"Test accuracy: {100*correct/total:.2f}%")

def save_raw(fname, tensor):
    tensor.cpu().contiguous().detach().numpy().astype("f").tofile(fname)

save_raw("conv1.w", net.conv1.weight.permute(0,2,3,1))
save_raw("conv1.b", net.conv1.bias)

save_raw("fc.w",  net.fc.weight)

perm = []
for h in range(14):
    for w in range(14):
        for c in range(8):
            perm.append(c*14*14 + h*14 + w)

fc_w_reordered = net.fc.weight[:, perm]
(
    fc_w_reordered
    .cpu()
    .detach()
    .astype("f")
    .tofile("fc.w")
)
print("fc.w rewritten in H‑W‑C order")

save_raw("fc.b",  net.fc.bias)

print("Weights written: conv1.w/.b  fc.w/.b")
