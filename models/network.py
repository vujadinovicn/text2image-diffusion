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
        frequencies = torch.exp(-math.log(self.max_period) / (half_dim - 1) * 
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
    
class AttentionBlock(nn.Module):
    def __init__(self, dim, groups=8):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, dim)
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        # Initialize weights to zero so the block starts as identity
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        input_x =x
        x = self.group_norm(x)

        q = self.q_proj(x).view(b, c, h * w).permute(0, 2, 1)
        k = self.k_proj(x).view(b, c, h * w)
        v = self.v_proj(x).view(b, c, h * w).permute(0, 2, 1)

        attn_weights = (q @ k / math.sqrt(c)).softmax(dim=-1) @ v
        attn_weights = attn_weights.permute(0, 2, 1).view(b, c, h, w)
        attn_out = self.out_proj(attn_weights)

        return attn_out + input_x
    
class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)
    
class Upsample(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        x = self.conv(x)
        return x
    
class UNet(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            base_channels=64,
            num_res_blocks=2,
            channel_mults=(1, 2, 4, 8),
            attention_resolutions=(16,),
            img_resolution=28,
            groups=8,
            dropout=0.0
        ):
        super().__init__()

        time_emb_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.in_proj = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        current_channels = base_channels
        current_img_resolution = img_resolution
        skip_channels = [current_channels]

        self.downs = nn.ModuleList()
        for i, mult in enumerate(channel_mults):
            for _ in range(num_res_blocks):
                self.downs.append(ResBlock(current_channels, base_channels * mult, time_emb_dim, groups=groups, dropout=dropout))
                current_channels = base_channels * mult
                if current_img_resolution in attention_resolutions:
                    self.downs.append(AttentionBlock(current_channels))
                skip_channels.append(current_channels)
            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(current_channels))
                current_img_resolution //= 2
                skip_channels.append(current_channels)


        self.bottleneck = nn.ModuleList([
            ResBlock(current_channels, current_channels, time_emb_dim, groups=groups, dropout=dropout),
            AttentionBlock(current_channels),
            ResBlock(current_channels, current_channels, time_emb_dim, groups=groups, dropout=dropout)
        ])

        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            for _ in range(num_res_blocks + 1):
                self.ups.append(ResBlock(skip_channels.pop() + current_channels, base_channels * mult, time_emb_dim, groups=groups, dropout=dropout))
                current_channels = base_channels * mult
                if current_img_resolution in attention_resolutions:
                    self.ups.append(AttentionBlock(current_channels))
                
            if i != 0:
                self.ups.append(Upsample(current_channels))
                current_img_resolution *= 2

        self.conv_group_block = ConvGroupBlock(current_channels, out_channels, groups=groups, init_weights=True)

    def forward(self, x, t):
        h = self.in_proj(x)
        time_emb = self.time_embedding(t)

        skip_connections = [h]
        for module in self.downs:
            if isinstance(module, Downsample):
                h = module(h)
                skip_connections.append(h)
            else:
                if isinstance(module, ResBlock):
                    h = module(h, time_emb) 
                    skip_connections.append(h) 
                else:
                    h = module(h)
            
        for module in self.bottleneck:
            h = module(h, time_emb) if isinstance(module, ResBlock) else module(h)

        for i, module in enumerate(self.ups):
            if isinstance(module, Upsample):
                h = module(h)
            else: 
                if isinstance(module, ResBlock):
                    skip_h = skip_connections.pop()
                    h = torch.cat((h, skip_h), dim=1)
                    h = module(h, time_emb)
                else: 
                    h = module(h)

        h = self.conv_group_block(h)
        return h