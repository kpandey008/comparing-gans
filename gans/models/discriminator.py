class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_1 = nn.Conv2d(1, 64, 4, stride=2, padding=1, bias=False)
        torch.nn.init.normal_(self.conv_1.weight, std=0.02)
        self.bn_1 = nn.BatchNorm2d(64)

        self.conv_2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)
        torch.nn.init.normal_(self.conv_2.weight, std=0.02)
        self.bn_2 = nn.BatchNorm2d(128)

        self.conv_3 = nn.Conv2d(128, 1, 7, bias=False)

    def forward(self, input_batch):
        conv_1_out = self.conv_1(input_batch)
        conv_1_out = F.leaky_relu(conv_1_out, negative_slope=0.2)

        conv_2_out = F.leaky_relu(self.bn_2(self.conv_2(conv_1_out)), negative_slope=0.2)
        conv_2_out = F.dropout(conv_2_out, p=0.3, training=True)

        conv_3_out = F.leaky_relu(self.conv_3(conv_2_out))
        output = torch.sigmoid(conv_3_out)
        output = torch.reshape(output, (-1, 1))
        return output

    def selective_forward(self, name, input_batch):
        if name == 'conv_1':
            output = F.leaky_relu(self.conv_1(input_batch), negative_slope=0.2)
            return output
        elif name == 'conv_2':
            output = F.leaky_relu(self.conv_1(input_batch), negative_slope=0.2)
            output = F.leaky_relu(self.conv_2(output), negative_slope=0.2)
            output = F.dropout(output, p=0.3, training=True)
            return output
        else:
            raise ValueError('Invalid module name')