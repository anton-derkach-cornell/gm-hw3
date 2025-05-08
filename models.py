import torch
import torch.nn.functional as F

class Block(torch.nn.Module):
    def __init__(self, filters):
        super(Block, self).__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(filters, filters, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(filters), torch.nn.ReLU(),
            torch.nn.Conv2d(filters, filters, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(filters))

    def forward(self, x):
        return F.relu(x + self.block(x))

class DBlock(torch.nn.Module):
    def __init__(self, filters, stride=1):
        super(DBlock, self).__init__()
        self.stride=stride

        # No BatchNorm
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(filters, filters, 3, padding=1, bias=False), torch.nn.ReLU(),
            torch.nn.Conv2d(filters, filters, 3, padding=1, stride=stride, bias=False))

    def forward(self, x):
        return F.relu(x[:,:,::self.stride,::self.stride] + self.block(x))

class Upsample(torch.nn.Module):
    def __init__(self, fin, fout, factor):
        super(Upsample, self).__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=False),
            torch.nn.Conv2d(fin, fout, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(fout), torch.nn.ReLU())

    def forward(self, x):
        return self.block(x)

class Generator(torch.nn.Module):
    def __init__(self, seed_size, capacity=128):
        super(Generator, self).__init__()
        self.capacity = capacity

        # Previous code for MNIST
        # self.embed = torch.nn.Linear(seed_size, capacity*7*7, bias=False)

        # Changed from capacity*7*7 to capacity*8*8 for 32x32 output for CIFAR-10
        self.embed = torch.nn.Linear(seed_size, capacity*8*8, bias=False)

        self.resnet = torch.nn.ModuleList()

        # Previous code for MNIST
        # for i in range(3): self.resnet.append(Block(capacity))

        # Increased from 3 to 9 residual blocks for CIFAR-10 images
        for i in range(9): 
            self.resnet.append(Block(capacity))
        self.resnet.append(Upsample(capacity, capacity, 4))

        # Previous code for MNIST
        # self.image = torch.nn.Conv2d(capacity, 1, 3, padding=1, bias=True)
        # self.bias = torch.nn.Parameter(torch.Tensor(1,28,28))

        # Changed output from 1 channel to 3 channels for RGB for CIFAR-10
        self.image = torch.nn.Conv2d(capacity, 3, 3, padding=1, bias=True)
        # Adjusted bias for 3x32x32 for CIFAR-10
        self.bias = torch.nn.Parameter(torch.Tensor(3, 32, 32))

        for name, parm in self.named_parameters():
            if name.endswith('weight'): torch.nn.init.normal_(parm, 0, .05)
            if name.endswith('bias'): torch.nn.init.constant_(parm, 0.0)

    def forward(self, s):
        # Previous code for MNIST
        # zx = F.relu(self.embed(s).view(-1,self.capacity,7,7))

        # Adjusted reshape for 8x8 spatial size for CIFAR-10
        zx = F.relu(self.embed(s).view(-1, self.capacity, 8, 8))

        for layer in self.resnet: zx = layer(zx)
        return torch.sigmoid(self.image(zx) + self.bias[None,:,:,:])

class Discriminator(torch.nn.Module):
    def __init__(self, capacity=128, weight_scale=.01):
        super(Discriminator, self).__init__()
        self.capacity = capacity

        # Previous code for MNIST
        # self.embed = torch.nn.Conv2d(1, capacity, 3, padding=1, bias=False)

        # Changed input from 1 channel to 3 channels for RGB for CIFAR-10
        self.embed = torch.nn.Conv2d(3, capacity, 3, padding=1, bias=False)

        self.resnet = torch.nn.ModuleList()
        self.resnet.append(DBlock(capacity, stride=4))
        for i in range(3): self.resnet.append(DBlock(capacity))

        self.out = torch.nn.Linear(capacity, 1, bias=True)

        for name, parm in self.named_parameters():
            if name.endswith('weight'): torch.nn.init.normal_(parm, 0, .05)
            if name.endswith('bias'): torch.nn.init.constant_(parm, 0.0)

    def forward(self, x):
        zx = F.relu(self.embed(x))
        for layer in self.resnet: zx = layer(zx)
        return self.out(zx.sum(dim=(2,3)))
