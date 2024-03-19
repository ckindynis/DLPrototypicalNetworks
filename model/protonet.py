from torch import nn
from typing import Optional

class ConvBlock(nn.module):
    def __init__(self, in_channels, out_channels: int, kernel_size: int):        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
    
    def forward(self, x):
        out = self.conv_block(x)
        return out


class ProtoNetEncoder(nn.module):
    def __init__(self, in_channels, out_channels: int, kernel_size, num_stacks: int, original_embedding_size: Optional[int] = None, new_embedding_size: Optional[int] = None):
        layers = []

        layers.append(ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size))

        for _ in range(num_stacks - 1):
            layers.append(ConvBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size))
        
        self.encoding_block = nn.Sequential(*layers)

        self.embedding_layer = None
        if original_embedding_size:
            self.embedding_layer = nn.Linear(original_embedding_size, new_embedding_size)
        
    def forward(self, x):
        embedding = self.encoding_block(x)

        embedding = embedding.view(embedding.size(0), -1)
        if self.embedding_layer:
            embedding = self.embedding_layer(embedding)
        
        return embedding
        
