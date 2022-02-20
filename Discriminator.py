from torch import nn
from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, Linear, CrossEntropyLoss, Dropout, LeakyReLU, \
    ConvTranspose2d


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        def discrimanator(inputSize, outputSize, Kernel=3, stride=1, padding=1, normalization=True):
            if normalization:
                self.encoder = Sequential(
                    Conv2d(inputSize, outputSize, Kernel, stride, padding),
                    LeakyReLU(),
                    nn.BatchNorm2d(outputSize)) #?? instanceNorm2d
                # 5 pool
            else:
                self.encoder = Sequential(
                    Conv2d(inputSize, outputSize, Kernel, stride, padding),
                    LeakyReLU())