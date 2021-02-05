"""
@Author: zhangluoyang
@E-mail: 55058629@qq.com
@Time:  12月 21, 2020
"""
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader
from model.deprecation.gpt import ImageGPT
from transformers.transformers import Quantize

centroids = np.load("mnist_centroids.npy")
transform = T.Compose([T.ToTensor(),
                       Quantize(centroids)])
minist = datasets.MNIST("data_set", train=True, download=True, transform=transform)

if __name__ == "__main__":
    batch_size = 32
    num_pixels = 28
    num_vocab = 16
    num_classes = 10
    num_heads = 2
    num_layers = 8
    learning_rate = 0.003
    epoch = 25000
    classify = True

    image_gpt = ImageGPT(num_pixels=num_pixels,
                         num_vocab=num_vocab,
                         num_classes=num_classes,
                         num_heads=num_heads,
                         num_layers=num_layers,
                         classify=classify)
    image_gpt = image_gpt.to("cuda:0")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(image_gpt.parameters(), lr=learning_rate, weight_decay=1e-5)
    for e in range(epoch):
        data_loader = DataLoader(minist, shuffle=True, batch_size=batch_size, num_workers=0)
        for i, (bx, by) in enumerate(data_loader):
            bx = bx.transpose(0, 1).contiguous()
            bx = bx.to("cuda:0")
            by = by.to("cuda:0")
            optimizer.zero_grad()
            p_cls, p_gen = image_gpt(bx, True)
            loss_cls = criterion(p_cls, by)
            loss_gen = criterion(p_gen.view(-1, p_gen.size(-1)), bx.view(-1))
            loss = loss_cls + loss_gen
            loss.backward()
            optimizer.step()
            print("e:{0}, loss_cls:{1}, loss_gen:{2}".format(e, loss.to("cpu").item(), loss_cls.to("cpu").item(),
                                                             loss_gen.to("cpu").item()))
