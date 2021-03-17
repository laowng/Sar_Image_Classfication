from module import *
from module.enet import ENet
class CNN(nn.Module):
    """
    Foo model
    """
    def __init__(self,num_class):
        super(CNN, self).__init__()
        self.conv1 = C_conv2d(6, 64, 3, 1, 1)
        self.lre1=C_LeakyReLU()
        self.conv2 = C_conv2d(64, 8, 3, 1, 1)
        self.enet=ENet(8,num_class)

    def forward(self, complex):
        complex = self.conv1(complex)
        complex = self.lre1(complex)
        complex = self.conv2(complex)
        input=complex.real
        out=self.enet(input)
        return out


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from data import Dataset
    cnn=CNN()


