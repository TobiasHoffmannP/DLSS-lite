import torch.nn as nn

class SimpleSR(nn.Module):
    def __init__(self, scale=2, num_channels=64):
        super().__init__()
        self.scale = scale

        self.head = nn.Conv2d(
            in_channels=3, # RGB
            out_channels=num_channels, # 64
            kernel_size=3, # 3x3
            stride=1, 
            padding=1
        )

        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(
                inplace=True
            )
        )

        self.upscale = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels * scale * scale,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.PixelShuffle(
                scale
            )
        )

        self.tail = nn.Conv2d(
            in_channels=num_channels,
            out_channels=3,
            kernel_size=3, 
            stride=1,
            padding=1
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x) + x 
        x = self.upscale(x)
        x = self.tail(x)
        return x

