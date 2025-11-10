import torch
import math
import torch.nn as nn

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps):
        half_dim = self.dim // 2
        frequencies = torch.exp(math.log(self.max_period) / (half_dim - 1) * 
                                torch.arange(half_dim)).to(timesteps.device)
        embeddings = timesteps[:, None] * frequencies[None, :]
        embeddings = torch.cat((torch.cos(embeddings), torch.sin(embeddings)), dim=-1)
        
        if self.dim % 2 == 1:
            embeddings = torch.cat((embeddings, torch.zeros_like(embeddings[:, :1])), dim=-1)
        
        return embeddings
    

class ConvGroupBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8, dropout=0.0, init_weights=False):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, in_channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Initialize weights so the block starts close to identity
        if init_weights:
            nn.init.zeros_(self.conv.weight)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        h = self.group_norm(x)
        h = self.activation(h)
        if self.dropout.p > 0:
            h = self.dropout(h)
        h = self.conv(h)
        return h
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8, dropout=0.0):
        super().__init__()
        self.conv_group_block1 = ConvGroupBlock(in_channels, out_channels, groups)
        self.conv_group_block2 = ConvGroupBlock(out_channels, out_channels, groups, dropout, init_weights=True)
        self.time_emb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.skip_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        h = self.conv_group_block1(x)
        time_emb = self.time_emb_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        h = self.conv_group_block2(h)

        return h + self.skip_connection(x)
