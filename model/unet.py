import torch
from torch import nn
import torch.nn.functional as F

'''

Downsampling:
32 x 32
16 x 16
8 x 8
4 x 4

2 convolutional residual blocks at each resolution
At 16 x 16 resolution, add attention block between convolutional blocks
Sinusoidal positional embeddings for time steps
'''

def conv3x3_halfsize(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

def conv3x3_samesize(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

def conv3x3_doublesize(in_channels, out_channels):
    # return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
    return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                         nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))

def conv1x1_samesize(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

def sinsoidal_embedding(timesteps, dim=256):
    # timesteps: [B]
    device = timesteps.device
    half_dim = dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = emb.to(device)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb  # [B, dim]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, dropout=0.1):
        super().__init__()

        self.activation = nn.SiLU()
        self.conv1 = conv3x3_samesize(in_channels, out_channels)
        self.conv2 = conv3x3_samesize(out_channels, out_channels)
        self.skip_conv = conv3x3_samesize(in_channels, out_channels) 

        self.group_norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.group_norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.t_mlp = nn.Linear(t_emb_dim, out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, t_emb):
        h = x                       # h: [B, C_in, H, W]

        h = self.group_norm1(h)     # h: [B, C_in, H, W]
        h = self.activation(h)      # h: [B, C_in, H, W]   
        h = self.conv1(h)           # h: [B, C_out, H, W]

        t_emb = self.activation(t_emb)      # t_emb: [B, T_dim]
        t_emb = self.t_mlp(t_emb)           # t_emb: [B, C_out]   

        h = h + t_emb[:, :, None, None]     # h: [B, C_out, H, W]
        h = self.dropout(h)                 # h: [B, C_out, H, W]

        h = self.group_norm2(h)             # h: [B, C_out, H, W]
        h = self.activation(h)              # h: [B, C_out, H, W]
        h = self.conv2(h)                   # h: [B, C_out, H, W]

        x = self.skip_conv(x)               # x: [B, C_out, H, W]
        return x + h                        # out: [B, C_out, H, W]

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.q_conv = conv1x1_samesize(in_channels, in_channels)
        self.k_conv = conv1x1_samesize(in_channels, in_channels)
        self.v_conv = conv1x1_samesize(in_channels, in_channels)

        self.out_conv = conv1x1_samesize(in_channels, in_channels)
    
    def forward(self, x, t_emb=None): # adding t_emb for having the same input arguments as ResidualBlock
        B, C, H, W = x.shape
        h = self.group_norm(x)              # h: [B, C, H, W]

        q = self.q_conv(h)                  # q: [B, C, H, W]
        k = self.k_conv(h)                  # k: [B, C, H, W]
        v = self.v_conv(h)                  # v: [B, C, H, W]

        q = q.view(B, C, H * W).permute(0, 2, 1)   # q: [B, H*W, C]
        k = k.view(B, C, H * W)                     # k: [B, C, H*W]
        v = v.view(B, C, H * W).permute(0, 2, 1)    # v: [B, H*W, C]

        attention_matrix = torch.bmm(q,k)/(C ** 0.5)                    # attention_matrix: [B, H*W, H*W] 
        attention_weights = torch.softmax(attention_matrix, dim=-1)     # attention_weights: [B, H*W, H*W]
        h = torch.bmm(attention_weights, v)                             # h: [B, H*W, H*W] x [B, H*W, C] = [B, H*W, C]

        h = h.permute(0, 2, 1).view(B, C, H, W)    # h: [B, C, H, W]
        h = self.out_conv(h)                       # h: [B, C, H, W]

        return x + h                               # out: [B, C, H, W]
    
class UNet(nn.Module):
    def __init__(self, 
                 resolutions_list = [32, 16, 8, 4],
                 in_channels = 1,
                 out_channel_multipliers = [1, 2, 4, 8],
                 starting_channels = 64,
                 num_resnet_blocks = 2,
                 attention_resolutions = [16],
                 T = 10,
                 T_dim = 256):
        """
        UNet architecture for Diffusion Models
        self.downsample_resnet_layers: [ [layer1, layer2..], [layer1, layer2..], ... ] 
        self.upsample_resnet_layers: [ [layer1, layer2..], [layer1, layer2..], ... ]
        len(self.downsample_resnet_layers) == len(self.upsample_resnet_layers) == len(resolutions_list)
        """
        
        super().__init__()
        self.T_dim = T_dim
        self.resolutions_list = resolutions_list
        self.in_channels = in_channels
        self.out_channel_multipliers = out_channel_multipliers
        self.num_resnet_blocks = num_resnet_blocks
        self.attention_resolutions = attention_resolutions

        # self.t_embedding_layer = nn.Embedding(T, T_dim)

        self.input_conv = conv3x3_samesize(in_channels, starting_channels)

        # Downsampling layers
        self.downsample_resnet_layers = nn.ModuleList()
        for i in range(len(resolutions_list)):

            # current resolution, current out channels
            res = resolutions_list[i]
            out_channels = out_channel_multipliers[i]

            if i == 0:
                in_channel = starting_channels
            else:
                in_channel = starting_channels * out_channel_multipliers[i-1]

            resnet_per_resolution = nn.ModuleList()
            for j in range(num_resnet_blocks):

                # Residual Block
                layer = ResidualBlock(
                    in_channels = in_channel if j == 0 else starting_channels * out_channels,
                    out_channels = starting_channels * out_channels,
                    t_emb_dim = T_dim
                )
                resnet_per_resolution.append(layer)

                # Attention Block
                if res in attention_resolutions:
                    attn_layer = AttentionBlock(
                        in_channels = starting_channels * out_channels
                    )
                    resnet_per_resolution.append(attn_layer)
                
            
            # Append all blocks for this resolution 
            self.downsample_resnet_layers.append(resnet_per_resolution)

        # downsampling layers
        self.downsize_layers = nn.ModuleList()
        for i in range(len(resolutions_list) - 1):
            in_ch = starting_channels * out_channel_multipliers[i]
            out_ch = starting_channels * out_channel_multipliers[i]
            downsample_layer = conv3x3_halfsize(in_ch, out_ch)
            self.downsize_layers.append(downsample_layer)

        
        # Middle Layers
        self.mid_resnet1 = ResidualBlock(
            in_channels = starting_channels * out_channel_multipliers[-1],
            out_channels = starting_channels * out_channel_multipliers[-1],
            t_emb_dim = T_dim
        )
        self.mid_attn = AttentionBlock(
            in_channels = starting_channels * out_channel_multipliers[-1]
        )
        self.mid_resnet2 = ResidualBlock(
            in_channels = starting_channels * out_channel_multipliers[-1],
            out_channels = starting_channels * out_channel_multipliers[-1],
            t_emb_dim = T_dim
        )

        # Upsampling layers
        self.upsample_resnet_layers = nn.ModuleList()
        for i in reversed(range(len(resolutions_list))): #i=3,2,1,0

            # current resolution, current out channels
            res = resolutions_list[i]

            in_channel = starting_channels* out_channel_multipliers[i]* 2  # due to skip connection

            resnet_per_resolution = nn.ModuleList()
            for j in range(num_resnet_blocks):
                
                if j==num_resnet_blocks-1:
                    out_channels = out_channel_multipliers[i-1] if i>0 else out_channel_multipliers[0] # change out_channels for last block in this resolution
                else:
                    out_channels = out_channel_multipliers[i]
                    
                # Attention Block
                if res in attention_resolutions:
                    attn_layer = AttentionBlock(
                        in_channels = starting_channels * out_channel_multipliers[i]
                    )
                    resnet_per_resolution.append(attn_layer)

                # Residual Block
                layer = ResidualBlock(
                    in_channels = in_channel,
                    out_channels = starting_channels * out_channels,
                    t_emb_dim = T_dim
                )
                resnet_per_resolution.append(layer)
                
            
            # Append all blocks for this resolution 
            self.upsample_resnet_layers.append(resnet_per_resolution)

        # upsampling layers
        self.upsize_layers = nn.ModuleList()
        for i in reversed(range(len(resolutions_list) - 1)): 
            in_ch = starting_channels * out_channel_multipliers[i] * 2  # due to skip connection
            out_ch = starting_channels * out_channel_multipliers[i]
            upsample_layer = conv3x3_doublesize(in_ch, out_ch)
            self.upsize_layers.append(upsample_layer)

        # output layers
        self.output_group_norm = nn.GroupNorm(num_groups=32, num_channels=starting_channels*out_channel_multipliers[0])
        self.output_activation = nn.SiLU()
        self.output_conv = conv3x3_samesize(starting_channels*out_channel_multipliers[0], in_channels)

    def forward(self, x, t):
        # x: [B, C_in, H, W]
        # t: [B]

        # Time embedding
        # t_emb = self.t_embedding_layer(t)    # t_emb: [B, T_dim]
        t_emb = sinsoidal_embedding(t, dim=self.T_dim)  # t_emb: [B, T_dim]

        # Initial Convolution
        h = self.input_conv(x)               # h: [B, C_start, H, W]

        # Downsampling 
        hs = []
        for i in range(len(self.resolutions_list)):
            resnet_blocks = self.downsample_resnet_layers[i]
            for layer in resnet_blocks:
                h = layer(h, t_emb)            # h: [B, C, H, W]
                if self.resolutions_list[i] in self.attention_resolutions:
                    if isinstance(layer, ResidualBlock): # append only after attention block
                        hs.append(h)                   
                else:
                    hs.append(h)                   # storing for skip connections

            # Downsampling Layer (except for last resolution)
            if i != len(self.resolutions_list) - 1:
                h = self.downsize_layers[i](h)  # h: [B, C, H/2, W/2]
                hs.append(h)                    # storing for skip connections
                        
        # Middle Layers
        h = self.mid_resnet1(h, t_emb)       # h: [B, C, H, W]
        h = self.mid_attn(h)                 # h: [B, C, H, W]
        h = self.mid_resnet2(h, t_emb)       # h: [B, C, H, W]

        # Upsampling
        for n,i in enumerate(reversed(range(len(self.resolutions_list)))):
            resnet_blocks = self.upsample_resnet_layers[n]
            for layer in resnet_blocks:
                # Skip Connection
                if isinstance(layer, ResidualBlock):
                    h_skip = hs.pop()
                    h = torch.cat([h, h_skip], dim=1)   # h: [B, 2C, H, W]
                    h = layer(h, t_emb)                  # h: [B, C, H, W]
                else:
                    h = layer(h)                        # h: [B, C, H, W]

            # Upsampling Layer (except for last resolution)
            if i != 0:
                h_skip = hs.pop()
                h = torch.cat([h, h_skip], dim=1)     # h: [B, 2C, H, W]
                h = self.upsize_layers[n](h)        # h: [B, C, 2H, 2W]
        
        # Output layers
        h = self.output_group_norm(h)          # h: [B, C_start, H, W]
        h = self.output_activation(h)          # h: [B, C_start, H, W]
        h = self.output_conv(h)                # h: [B, C_in, H, W]

        return F.tanh(h)                               # out: [B, C_in, H, W]

# Testing the UNet model
if __name__ == "__main__":
    model = UNet()
    x = torch.randn(4, 1, 32, 32)  # batch of 4, 1 channel, 32x32 images
    t = torch.randint(0, 1000, (4,))  # batch of 4 time steps
    print(t.shape)
    out = model(x, t)
    print(out.shape)  # should be [4, 1, 32, 32]

# TO DO: remove print statements after debugging
# try denoising autoencoder to verify the model works