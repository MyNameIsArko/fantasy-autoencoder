from torch import nn
import torch
from utils import reparameterize


class ResidualBlock(nn.Module):
    def __init__(self, inc=64, outc=64):
        super(ResidualBlock, self).__init__()

        if inc != outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0,
                                         bias=False)
        else:
            self.conv_expand = None

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(outc),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(outc)
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        output = self.conv(x)
        output = self.relu(torch.add(output, identity_data))
        return output


class Encoder(nn.Module):
    def __init__(self, cdim=3, zdim=512, channels=(64, 128, 256, 512, 512, 512), image_size=256):
        super(Encoder, self).__init__()
        self.zdim = zdim
        self.cdim = cdim
        self.image_size = image_size
        cc = channels[0]
        layers = [
            nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
            nn.BatchNorm2d(cc),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        ]

        sz = image_size // 2
        for ch in channels[1:]:
            layers.append(ResidualBlock(cc, ch))
            layers.append(nn.AvgPool2d(2))
            cc, sz = ch, sz // 2

        layers.append(ResidualBlock(cc, cc))
        self.main = nn.Sequential(*layers)
        self.conv_output_size = self.calc_conv_output_size()
        num_fc_features = torch.zeros(self.conv_output_size).view(-1).shape[0]
        # print("conv shape: ", self.conv_output_size)
        # print("num fc features: ", num_fc_features)
        self.fc = nn.Linear(num_fc_features, 2 * zdim)

    def calc_conv_output_size(self):
        dummy_input = torch.zeros(1, self.cdim, self.image_size, self.image_size)
        dummy_input = self.main(dummy_input)
        return dummy_input[0].shape

    def forward(self, x):
        y = self.main(x).view(x.size(0), -1)
        y = self.fc(y)
        mu, logvar = y.chunk(2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, cdim=3, zdim=512, channels=(64, 128, 256, 512, 512, 512), image_size=256,
                 conv_input_size=None):
        super(Decoder, self).__init__()
        self.cdim = cdim
        self.image_size = image_size
        cc = channels[-1]
        self.conv_input_size = conv_input_size
        if conv_input_size is None:
            num_fc_features = cc * 4 * 4
        else:
            num_fc_features = torch.zeros(self.conv_input_size).view(-1).shape[0]
        self.fc = nn.Sequential(
            nn.Linear(zdim, num_fc_features),
            nn.ReLU(True),
        )

        sz = 4

        self.main = nn.Sequential()
        layers = []
        for ch in channels[::-1]:
            layers.append(ResidualBlock(cc, ch))
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz * 2

        layers.append(ResidualBlock(cc, cc))
        layers.append(nn.Conv2d(cc, cdim, 5, 1, 2))

        self.main = nn.Sequential(*layers)

    def forward(self, z):
        z = z.view(z.size(0), -1)
        y = self.fc(z)
        y = y.view(z.size(0), *self.conv_input_size)
        y = self.main(y)
        return y


class SoftIntroVAE(nn.Module):
    def __init__(self, cdim=3, zdim=512, channels=(64, 128, 256, 512, 512, 512), image_size=256,
                 cond_dim=10):
        super(SoftIntroVAE, self).__init__()

        self.zdim = zdim
        self.cond_dim = cond_dim

        self.encoder = Encoder(cdim, zdim, channels, image_size)

        self.decoder = Decoder(cdim, zdim, channels, image_size,
                               conv_input_size=self.encoder.conv_output_size)

    def forward(self, x, deterministic=False):
        mu, logvar = self.encode(x)
        if deterministic:
            z = mu
        else:
            z = reparameterize(mu, logvar)
        y = self.decode(z)
        return mu, logvar, z, y

    def sample(self, z):
        y = self.decode(z)
        return y

    def sample_with_noise(self, num_samples=1, device=torch.device("cpu")):
        z = torch.randn(num_samples, self.zdim).to(device)
        return self.decode(z)

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z):
        y = self.decoder(z)
        return y
