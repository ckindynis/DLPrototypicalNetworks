from torch import nn
from typing import Optional


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels: int,
        conv_kernel_size: int,
        max_pool_kernel: int,
    ):  
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=conv_kernel_size,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=max_pool_kernel),
        )

    def forward(self, x):
        out = self.conv_block(x)
        return out


class ProtoNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels: int,
        conv_kernel_size,
        max_pool_kernel: int,
        num_conv_layers: int,
        original_embedding_size: Optional[int] = None,
        new_embedding_size: Optional[int] = None,
    ):  
        super(ProtoNetEncoder, self).__init__()

        layers = []

        layers.append(
            ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                conv_kernel_size=conv_kernel_size,
                max_pool_kernel=max_pool_kernel,
            )
        )

        for _ in range(num_conv_layers - 1):
            layers.append(
                ConvBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    conv_kernel_size=conv_kernel_size,
                    max_pool_kernel=max_pool_kernel,
                )
            )

        self.encoding_block = nn.Sequential(*layers)

        self.embedding_layer = None
        if new_embedding_size:
            self.embedding_layer = nn.Linear(
                original_embedding_size, new_embedding_size
            )

    def forward(self, x):
        embedding = self.encoding_block(x)

        embedding = embedding.view(embedding.size(0), -1)
        if self.embedding_layer:
            embedding = self.embedding_layer(embedding)

        return embedding
