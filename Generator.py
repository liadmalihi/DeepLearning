import torch.nn as nn
from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, Linear, CrossEntropyLoss, Dropout, LeakyReLU, \
    ConvTranspose2d


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def Encoder(inputSize, outputSize, Kernel=4, stride=1, padding=1, normalization=True):
            if normalization:
                self.encoder = Sequential(
                    Conv2d(inputSize, outputSize,Kernel , stride, padding),
                    LeakyReLU(),
                    nn.BatchNorm2d(outputSize))
                    # 5 pool
            else:
                self.encoder = Sequential(
                    Conv2d(inputSize, outputSize, Kernel, stride, padding),
                    LeakyReLU())
        def Fc(inputSize, outputSize=4000, kernel=1):
            self.fc = nn.Conv2d(inputSize, outputSize, kernel)
        def Decoder(inputSize, outputSize, Kernel=4, stride=1, padding=1):
            self.decoder = Sequential(
                ConvTranspose2d(inputSize, outputSize, Kernel, stride, padding),
                ReLU())
                # nn.BatchNorm2d(outputSize)) - without batch?
        def forward(self, x):
            # self.encoder(128, 128, False)
            self.encoder(64, 64)
            self.encoder(64, 128)
            self.encoder(128, 256)
            self.encoder(256, 512)
            self.fc(512)
            self.decoder(4000, 512)
            self.decoder(512, 256)
            self.decoder(256, 128)
            self.decoder(128, 64)
            # self.fc(64, 3)
            # nn.Tanh
