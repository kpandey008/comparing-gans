import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, code_size):
        super(Generator, self).__init__()
        self.code_size = code_size

        # Define the model here
        self.deconv_1 = nn.ConvTranspose2d(code_size, 256, 4, stride=2, bias=False)
        torch.nn.init.normal_(self.deconv_1.weight, std=0.02)
        self.bn_1 = nn.BatchNorm2d(256)

        self.deconv_2 = nn.ConvTranspose2d(256, 128, 4, bias=False)
        torch.nn.init.normal_(self.deconv_2.weight, std=0.02)
        self.bn_2 = nn.BatchNorm2d(128)

        self.deconv_3 = nn.ConvTranspose2d(128, 64, 4, padding=1, stride=2, bias=False)
        torch.nn.init.normal_(self.deconv_3.weight, std=0.02)
        self.bn_3 = nn.BatchNorm2d(64)

        self.deconv_4 = nn.ConvTranspose2d(64, 1, 4, padding=1, stride=2, bias=False)
        torch.nn.init.normal_(self.deconv_4.weight, std=0.02)

    def forward(self, z_batch):
        # Convert the input noise vector into a (10 x 10) tensor
        z_batch = torch.reshape(z_batch, (-1, self.code_size, 1, 1))
        deconv_1_out = F.relu(self.bn_1(self.deconv_1(z_batch)))
        deconv_2_out = F.relu(self.bn_2(self.deconv_2(deconv_1_out)))

        deconv_3_out = F.relu(self.deconv_3(deconv_2_out))
        deconv_4_out = F.relu(self.deconv_4(deconv_3_out))
        output = torch.tanh(deconv_4_out)
        return output
