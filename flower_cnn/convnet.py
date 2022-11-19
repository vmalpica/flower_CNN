import torch
import torch.nn as nn



class ConvNetwork(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1)) #same convolution
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))

        self.mlp = nn.Sequential(
            nn.Linear(32*16*16, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.mlp(x)

        return x
        