import json
import torch
import torch.nn as nn

f = open("configs.json")
configs = json.load(f)


def _get_act_func():
    act_func = configs["model_properties"]["act_func"].lower()
    match act_func:
        case "relu":
            return nn.ReLU()
        case "lrelu":
            return nn.LeakyReLU()
        case "selu":
            return nn.SELU()
        case "elu":
            return nn.ELU()
        case "sigmoid":
            return nn.Sigmoid()
        case "tanh":
            return nn.Tanh()
        case "prelu":
            return nn.PReLU()
        case "rrelu":
            return nn.RReLU()
        case "celu":
            return nn.CELU()
        case "silu":
            return nn.SiLU()
        case default:
            raise NameError(
                "Activation function can only be from the list - relu, lrelu, selu, elu, sigmoid, tanh, prelu, rrelu, celu and silu."
            )


class _DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            _get_act_func(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            _get_act_func(),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNet
        for feature in features:
            self.downs.append(_DoubleConv(in_channels, feature))
            in_channels = feature
        self.bottleneck = _DoubleConv(features[-1], features[-1] * 2)

        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(_DoubleConv(feature * 2, feature))

        self.out = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Down part of UNet
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Up part of UNet
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(
                    x,
                    size=skip_connection.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.out(x)


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unet = UNet()
        self.fwd = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            _get_act_func(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.unet(x)
        return self.fwd(x)


class Discriminator(nn.Module):
    def __init__(self, num_channels=3, num_filters=64, num_classes=1):
        super(Discriminator, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, num_filters, kernel_size=4, stride=2, padding=1),
            _get_act_func(),
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters * 2),
            _get_act_func(),
            nn.Conv2d(
                num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_filters * 4),
            _get_act_func(),
        )

        self.adaptive_pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_layers = nn.Sequential(
            nn.Linear(num_filters * 4, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
