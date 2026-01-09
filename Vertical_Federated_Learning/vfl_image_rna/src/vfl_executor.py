import torch
import torch.nn as nn
from nvflare.apis.executor import Executor
from nvflare.apis.shareable import Shareable

import os
import matplotlib.pyplot as plt


#for dummy data 
import torch
from torch.utils.data import DataLoader, TensorDataset


# ---------- Models ----------
class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(8, 16)
        )

    def forward(self, x):
        return self.net(x)


class RNAModel(nn.Module):
    def __init__(self, gene_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gene_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

    def forward(self, x):
        return self.net(x)


class ServerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 1)

    def forward(self, h_img, h_rna):
        z = torch.cat([h_img, h_rna], dim=1)
        return self.fc(z)


# ---------- Executors ----------
class ImageExecutor(Executor):
    def __init__(self):
        super().__init__()

        # Dummy image data
        num_samples = 1000
        images = torch.randn(num_samples, 3, 28, 28)

        dataset = TensorDataset(images)
        self.loader = DataLoader(dataset, batch_size=32, shuffle=True)
        self.data_iter = iter(self.loader)

        self.model = ImageModel()
        self.opt = torch.optim.Adam(self.model.parameters(), 1e-3)

    def execute(self, task_name, shareable, fl_ctx):

        if task_name == "vfl_forward":
            try:
                (x,) = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.loader)
                (x,) = next(self.data_iter)

            h = self.model(x)
            h.requires_grad_()

            return Shareable({"h_img": h})

        elif task_name == "vfl_backward":
            grad = shareable["grad_img"]

            self.opt.zero_grad()
            shareable["h_img"].backward(grad)
            self.opt.step()

            return Shareable()


class RNAExecutor(Executor):
    def __init__(self, gene_dim=20000):
        super().__init__()

        num_samples = 1000
        rna_seq = torch.randn(num_samples, gene_dim)

        dataset = TensorDataset(rna_seq)
        self.loader = DataLoader(dataset, batch_size=32, shuffle=True)
        self.data_iter = iter(self.loader)

        self.model = RNAModel(gene_dim)
        self.opt = torch.optim.Adam(self.model.parameters(), 1e-3)

    def execute(self, task_name, shareable, fl_ctx):
        if task_name == "vfl_forward":
            try:
                (x,) = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.loader)
                (x,) = next(self.data_iter)

            h = self.model(x)
            h.requires_grad_()

            return Shareable({"h_rna": h})

        elif task_name == "vfl_backward":
            grad = shareable["grad_rna"]

            self.opt.zero_grad()
            self.model.backward(grad)
            shareable["h_rna"].backward(grad)
            self.opt.step()

            return Shareable()
