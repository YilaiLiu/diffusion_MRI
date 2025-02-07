# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SPADE(nn.Module):
    "根据正常归一化后的结果去计算对呀的β和μ从而进行仿射变换"
    def __init__(self, norm_nc, label_nc, kernel_size=3, norm_type='instance'):
        super().__init__()

        if norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm3d(norm_nc, affine=False)
        elif norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm3d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 64

        pw = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv3d(label_nc, nhidden, kernel_size=kernel_size, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv3d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)
        self.mlp_beta = nn.Conv3d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)

    def forward(self, x):
        normalized = self.param_free_norm(x)

        x = F.interpolate(x, size=x.size()[2:], mode='nearest')#没用？
        actv = self.mlp_shared(x)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out
    
class SPADE_Multimodal(nn.Module):
    def __init__(self, modalities, norm_nc, label_nc, kernel_size, norm_type='instance'):
        """label_nc是SPADE的输入ch数,norm_nc是输出channel数,默认是一样的"""
        super().__init__()
        self.spades = nn.ModuleDict({modality: SPADE(norm_nc, label_nc, kernel_size, norm_type) for modality in modalities})
        self.spades["unknown"]=SPADE(norm_nc, label_nc, kernel_size, norm_type) 
    def forward(self, x, modality):
        if modality in self.spades:
            x = self.spades[modality](x)
        else:
            raise ValueError('%s is not a recognized modality in SPADE_Multimodal' % modality)
        return x
def cond_Normalize(conditions,ch,kernel_size,norm_type='instance'):
    return SPADE_Multimodal(conditions,ch,ch,kernel_size,norm_type)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        # self.norm2 = Normalize(out_channels)
        # self.dropout = torch.nn.Dropout(dropout)
        # self.conv2 = torch.nn.Conv3d(out_channels,
        #                              out_channels,
        #                              kernel_size=3,
        #                              stride=1,
        #                              padding=1)
        # if self.in_channels != self.out_channels:
        #     if self.use_conv_shortcut:
        #         self.conv_shortcut = torch.nn.Conv3d(in_channels,
        #                                              out_channels,
        #                                              kernel_size=3,
        #                                              stride=1,
        #                                              padding=1)
        #     else:
        #         self.nin_shortcut = torch.nn.Conv3d(in_channels,
        #                                             out_channels,
        #                                             kernel_size=1,
        #                                             stride=1,
        #                                             padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        # if temb is not None:
        #     h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        # h = self.norm2(h)
        # h = nonlinearity(h)
        # h = self.dropout(h)
        # h = self.conv2(h)

        # if self.in_channels != self.out_channels:
        #     if self.use_conv_shortcut:
        #         x = self.conv_shortcut(x)
        #     else:
        #         x = self.nin_shortcut(x)

        return h
class cond_ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512,conditions=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = cond_Normalize(conditions=conditions,ch=in_channels,kernel_size=3)
        self.conv1 = torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        # self.norm2 = Normalize(out_channels)
        # self.dropout = torch.nn.Dropout(dropout)
        # self.conv2 = torch.nn.Conv3d(out_channels,
        #                              out_channels,
        #                              kernel_size=3,
        #                              stride=1,
        #                              padding=1)
        # if self.in_channels != self.out_channels:
        #     if self.use_conv_shortcut:
        #         self.conv_shortcut = torch.nn.Conv3d(in_channels,
        #                                              out_channels,
        #                                              kernel_size=3,
        #                                              stride=1,
        #                                              padding=1)
        #     else:
        #         self.nin_shortcut = torch.nn.Conv3d(in_channels,
        #                                             out_channels,
        #                                             kernel_size=1,
        #                                             stride=1,
        #                                             padding=0)

    def forward(self, x, cond):
        h = x
        h = self.norm1(h,cond)
        h = nonlinearity(h)
        h = self.conv1(h)

        # if temb is not None:
        #     h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        # h = self.norm2(h)
        # h = nonlinearity(h)
        # h = self.dropout(h)
        # h = self.conv2(h)

        # if self.in_channels != self.out_channels:
        #     if self.use_conv_shortcut:
        #         x = self.conv_shortcut(x)
        #     else:
        #         x = self.nin_shortcut(x)

        return h




class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w,d = q.shape
        q = q.reshape(b,c,h*w*d)
        q = q.permute(0,2,1)   # b,hwd,c
        k = k.reshape(b,c,h*w*d) # b,c,hwd
        w_ = torch.bmm(q,k)     # b,hwd,hwd    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w*d)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w,d)

        h_ = self.proj_out(h_)

        return x+h_


class Model(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, use_timestep=True):
        super().__init__()
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # downsampling
        self.conv_in = torch.nn.Conv3d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        # self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x, t=None):
        #assert x.shape[2] == x.shape[3] == self.resolution

        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        # h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv3d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            # attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                # if curr_res in attn_resolutions:
                #     attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            # down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        # self.mid.attn_1 = AttnBlock(block_in)
        # self.mid.block_2 = ResnetBlock(in_channels=block_in,
        #                                out_channels=block_in,
        #                                temb_channels=self.temb_ch,
        #                                dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x):
        #assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                # if len(self.down[i_level].attn) > 0:
                #     h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        # h = self.mid.attn_1(h)
        # h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv3d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        # self.mid.attn_1 = AttnBlock(block_in)
        # self.mid.block_2 = ResnetBlock(in_channels=block_in,
        #                                out_channels=block_in,
        #                                temb_channels=self.temb_ch,
        #                                dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            # attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                # if curr_res in attn_resolutions:
                #     attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            # up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        # h = self.mid.attn_1(h)
        # h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                # if len(self.up[i_level].attn) > 0:
                #     h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class cond_Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        conditions
        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv3d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = cond_ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        # self.mid.attn_1 = AttnBlock(block_in)
        # self.mid.block_2 = ResnetBlock(in_channels=block_in,
        #                                out_channels=block_in,
        #                                temb_channels=self.temb_ch,
        #                                dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            # attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(cond_ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                # if curr_res in attn_resolutions:
                #     attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            # up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = cond_Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z,condition):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        # h = self.mid.attn_1(h)
        # h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                # if len(self.up[i_level].attn) > 0:
                #     h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h,condition)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h





class VUNet(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 in_channels, c_channels,
                 resolution, z_channels, use_timestep=False, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # downsampling
        self.conv_in = torch.nn.Conv3d(c_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.z_in = torch.nn.Conv3d(z_channels,
                                    block_in,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=2*block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x, z):
        #assert x.shape[2] == x.shape[3] == self.resolution

        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        z = self.z_in(z)
        h = torch.cat((h,z),dim=1)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.model = nn.ModuleList([nn.Conv3d(in_channels, in_channels, 1),
                                     ResnetBlock(in_channels=in_channels,
                                                 out_channels=2 * in_channels,
                                                 temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=2 * in_channels,
                                                out_channels=4 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=4 * in_channels,
                                                out_channels=2 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     nn.Conv3d(2*in_channels, in_channels, 1),
                                     Upsample(in_channels, with_conv=True)])
        # end
        self.norm_out = Normalize(in_channels)
        self.conv_out = torch.nn.Conv3d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i in [1,2,3]:
                x = layer(x, None)
            else:
                x = layer(x)

        h = self.norm_out(x)
        h = nonlinearity(h)
        x = self.conv_out(h)
        return x


class UpsampleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, ch, num_res_blocks, resolution,
                 ch_mult=(2,2), dropout=0.0):
        super().__init__()
        # upsampling
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = in_channels
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.res_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            res_block = []
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            self.res_blocks.append(nn.ModuleList(res_block))
            if i_level != self.num_resolutions - 1:
                self.upsample_blocks.append(Upsample(block_in, True))
                curr_res = curr_res * 2

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # upsampling
        h = x
        for k, i_level in enumerate(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.res_blocks[i_level][i_block](h, None)
            if i_level != self.num_resolutions - 1:
                h = self.upsample_blocks[k](h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


# # pytorch_diffusion + derived encoder decoder
# import math
# import torch
# import torch.nn as nn
# import numpy as np
# from einops import rearrange


# def get_timestep_embedding(timesteps, embedding_dim):
#     """
#     This matches the implementation in Denoising Diffusion Probabilistic Models:
#     From Fairseq.
#     Build sinusoidal embeddings.
#     This matches the implementation in tensor2tensor, but differs slightly
#     from the description in Section 3.5 of "Attention Is All You Need".
#     """
#     assert len(timesteps.shape) == 1

#     half_dim = embedding_dim // 2
#     emb = math.log(10000) / (half_dim - 1)
#     emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
#     emb = emb.to(device=timesteps.device)
#     emb = timesteps.float()[:, None] * emb[None, :]
#     emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
#     if embedding_dim % 2 == 1:  # zero pad
#         emb = torch.nn.functional.pad(emb, (0,1,0,0))
#     return emb


# def nonlinearity(x):
#     # swish
#     return x*torch.sigmoid(x)


# def Normalize(in_channels):
#     return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


# class Upsample(nn.Module):
#     def __init__(self, in_channels, with_conv):
#         super().__init__()
#         self.with_conv = with_conv
#         if self.with_conv:
#             self.conv = torch.nn.Conv3d(in_channels,
#                                         in_channels,
#                                         kernel_size=3,
#                                         stride=1,
#                                         padding=1)

#     def forward(self, x):
#         x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
#         if self.with_conv:
#             x = self.conv(x)
#         return x


# class Downsample(nn.Module):
#     def __init__(self, in_channels, with_conv):
#         super().__init__()
#         self.with_conv = with_conv
#         if self.with_conv:
#             # no asymmetric padding in torch conv, must do it ourselves
#             self.conv = torch.nn.Conv3d(in_channels,
#                                         in_channels,
#                                         kernel_size=3,
#                                         stride=2,
#                                         padding=0)

#     def forward(self, x):
#         if self.with_conv:
#             pad = (0,1,0,1,0,1)
#             x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
#             x = self.conv(x)
#         else:
#             x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
#         return x


# class ResnetBlock(nn.Module):
#     def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
#                  dropout, temb_channels=512):
#         super().__init__()
#         self.in_channels = in_channels
#         out_channels = in_channels if out_channels is None else out_channels
#         self.out_channels = out_channels
#         self.use_conv_shortcut = conv_shortcut

#         self.norm1 = Normalize(in_channels)
#         self.conv1 = torch.nn.Conv3d(in_channels,
#                                      out_channels,
#                                      kernel_size=3,
#                                      stride=1,
#                                      padding=1)
#         if temb_channels > 0:
#             self.temb_proj = torch.nn.Linear(temb_channels,
#                                              out_channels)
#         self.norm2 = Normalize(out_channels)
#         self.dropout = torch.nn.Dropout(dropout)
#         self.conv2 = torch.nn.Conv3d(out_channels,
#                                      out_channels,
#                                      kernel_size=3,
#                                      stride=1,
#                                      padding=1)
#         if self.in_channels != self.out_channels:
#             if self.use_conv_shortcut:
#                 self.conv_shortcut = torch.nn.Conv3d(in_channels,
#                                                      out_channels,
#                                                      kernel_size=3,
#                                                      stride=1,
#                                                      padding=1)
#             else:
#                 self.nin_shortcut = torch.nn.Conv3d(in_channels,
#                                                     out_channels,
#                                                     kernel_size=1,
#                                                     stride=1,
#                                                     padding=0)

#     def forward(self, x, temb):
#         h = x
#         h = self.norm1(h)
#         h = nonlinearity(h)
#         h = self.conv1(h)

#         if temb is not None:
#             h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

#         h = self.norm2(h)
#         h = nonlinearity(h)
#         h = self.dropout(h)
#         h = self.conv2(h)

#         if self.in_channels != self.out_channels:
#             if self.use_conv_shortcut:
#                 x = self.conv_shortcut(x)
#             else:
#                 x = self.nin_shortcut(x)

#         return h
    

# class AttnBlock(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.in_channels = in_channels

#         self.norm = Normalize(in_channels)
#         self.q = torch.nn.Conv3d(in_channels,
#                                  in_channels,
#                                  kernel_size=1,
#                                  stride=1,
#                                  padding=0)
#         self.k = torch.nn.Conv3d(in_channels,
#                                  in_channels,
#                                  kernel_size=1,
#                                  stride=1,
#                                  padding=0)
#         self.v = torch.nn.Conv3d(in_channels,
#                                  in_channels,
#                                  kernel_size=1,
#                                  stride=1,
#                                  padding=0)
#         self.proj_out = torch.nn.Conv3d(in_channels,
#                                         in_channels,
#                                         kernel_size=1,
#                                         stride=1,
#                                         padding=0)

#     def forward(self, x):
#         h_ = x
#         h_ = self.norm(h_)
#         q = self.q(h_)
#         k = self.k(h_)
#         v = self.v(h_)

#         # compute attention
#         b,c,h,w,d = q.shape
#         q = q.reshape(b,c,h*w*d)
#         q = q.permute(0,2,1)     # b,hwd,c
#         k = k.reshape(b,c,h*w*d) # b,c,hwd
#         w_ = torch.bmm(q,k)     # b,hwd,hwd    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
#         w_ = w_ * (int(c)**(-0.5))
#         w_ = torch.nn.functional.softmax(w_, dim=2)

#         # attend to values
#         v = v.reshape(b,c,h*w*d)
#         w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
#         h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
#         h_ = h_.reshape(b,c,h,w,d)

#         h_ = self.proj_out(h_)

#         return x+h_
    
# class FactorizedAttnBlock(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.in_channels = in_channels

#         self.norm = Normalize(in_channels)
#         self.q = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
#         self.k = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
#         self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
#         self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

#     def factorized_attention(self, q, k, v, size):
#         b, c, s1, s2 = q.shape
#         q = q.reshape(b, c, s1*s2)
#         q = q.permute(0, 2, 1)
#         k = k.reshape(b, c, s1*s2)
#         w_ = torch.bmm(q, k)
#         w_ = w_ * (int(c)**(-0.5))
#         w_ = nn.functional.softmax(w_, dim=2)
#         v = v.reshape(b, c, s1*s2)
#         h_ = torch.bmm(v, w_.permute(0, 2, 1))
#         h_ = h_.reshape(b, c, s1, s2)
#         return h_

#     def forward(self, x):
#         h_ = self.norm(x)
#         b, c, h, w, d = h_.shape

#         q = self.q(h_).reshape(b*d, c, h, w)
#         k = self.k(h_).reshape(b*d, c, h, w)
#         v = self.v(h_).reshape(b*d, c, h, w)
#         h = self.factorized_attention(q, k, v, (h, w))
#         h = rearrange(h, '(b d) c h w -> b c h w d', b=b, d=d)

#         h = self.proj_out(h)
#         return x + h


# class Model(nn.Module):
#     def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
#                  attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
#                  resolution, use_timestep=True):
#         super().__init__()
#         self.ch = ch
#         self.temb_ch = self.ch*4
#         self.num_resolutions = len(ch_mult)
#         self.num_res_blocks = num_res_blocks
#         self.resolution = resolution
#         self.in_channels = in_channels

#         self.use_timestep = use_timestep
#         if self.use_timestep:
#             # timestep embedding
#             self.temb = nn.Module()
#             self.temb.dense = nn.ModuleList([
#                 torch.nn.Linear(self.ch,
#                                 self.temb_ch),
#                 torch.nn.Linear(self.temb_ch,
#                                 self.temb_ch),
#             ])

#         # downsampling
#         self.conv_in = torch.nn.Conv3d(in_channels,
#                                        self.ch,
#                                        kernel_size=3,
#                                        stride=1,
#                                        padding=1)

#         curr_res = resolution
#         in_ch_mult = (1,)+tuple(ch_mult)
#         self.down = nn.ModuleList()
#         for i_level in range(self.num_resolutions):
#             block = nn.ModuleList()
#             attn = nn.ModuleList()
#             block_in = ch*in_ch_mult[i_level]
#             block_out = ch*ch_mult[i_level]
#             for i_block in range(self.num_res_blocks):
#                 block.append(ResnetBlock(in_channels=block_in,
#                                          out_channels=block_out,
#                                          temb_channels=self.temb_ch,
#                                          dropout=dropout))
#                 block_in = block_out
#                 # if curr_res in attn_resolutions:
#                 #     attn.append(FactorizedAttnBlock(block_in))
#             down = nn.Module()
#             down.block = block
#             down.attn = attn
#             if i_level != self.num_resolutions-1:
#                 down.downsample = Downsample(block_in, resamp_with_conv)
#                 curr_res = curr_res // 2
#             self.down.append(down)

#         # middle
#         self.mid = nn.Module()
#         self.mid.block_1 = ResnetBlock(in_channels=block_in,
#                                        out_channels=block_in,
#                                        temb_channels=self.temb_ch,
#                                        dropout=dropout)
#         # self.mid.attn_1 = FactorizedAttnBlock(block_in)
#         self.mid.block_2 = ResnetBlock(in_channels=block_in,
#                                        out_channels=block_in,
#                                        temb_channels=self.temb_ch,
#                                        dropout=dropout)

#         # upsampling
#         self.up = nn.ModuleList()
#         for i_level in reversed(range(self.num_resolutions)):
#             block = nn.ModuleList()
#             attn = nn.ModuleList()
#             block_out = ch*ch_mult[i_level]
#             skip_in = ch*ch_mult[i_level]
#             for i_block in range(self.num_res_blocks+1):
#                 if i_block == self.num_res_blocks:
#                     skip_in = ch*in_ch_mult[i_level]
#                 block.append(ResnetBlock(in_channels=block_in+skip_in,
#                                          out_channels=block_out,
#                                          temb_channels=self.temb_ch,
#                                          dropout=dropout))
#                 block_in = block_out
#                 # if curr_res in attn_resolutions:
#                 #     attn.append(FactorizedAttnBlock(block_in))
#             up = nn.Module()
#             up.block = block
#             up.attn = attn
#             if i_level != 0:
#                 up.upsample = Upsample(block_in, resamp_with_conv)
#                 curr_res = curr_res * 2
#             self.up.insert(0, up) # prepend to get consistent order

#         # end
#         self.norm_out = Normalize(block_in)
#         self.conv_out = torch.nn.Conv3d(block_in,
#                                         out_ch,
#                                         kernel_size=3,
#                                         stride=1,
#                                         padding=1)


#     def forward(self, x, t=None):
#         #assert x.shape[2] == x.shape[3] == self.resolution

#         if self.use_timestep:
#             # timestep embedding
#             assert t is not None
#             temb = get_timestep_embedding(t, self.ch)
#             temb = self.temb.dense[0](temb)
#             temb = nonlinearity(temb)
#             temb = self.temb.dense[1](temb)
#         else:
#             temb = None

#         # downsampling
#         hs = [self.conv_in(x)]
#         for i_level in range(self.num_resolutions):
#             for i_block in range(self.num_res_blocks):
#                 h = self.down[i_level].block[i_block](hs[-1], temb)
#                 if len(self.down[i_level].attn) > 0:
#                     h = self.down[i_level].attn[i_block](h)
#                 hs.append(h)
#             if i_level != self.num_resolutions-1:
#                 hs.append(self.down[i_level].downsample(hs[-1]))

#         # middle
#         h = hs[-1]
#         h = self.mid.block_1(h, temb)
#         # h = self.mid.attn_1(h)
#         h = self.mid.block_2(h, temb)

#         # upsampling
#         for i_level in reversed(range(self.num_resolutions)):
#             for i_block in range(self.num_res_blocks+1):
#                 h = self.up[i_level].block[i_block](
#                     torch.cat([h, hs.pop()], dim=1), temb)
#                 if len(self.up[i_level].attn) > 0:
#                     h = self.up[i_level].attn[i_block](h)
#             if i_level != 0:
#                 h = self.up[i_level].upsample(h)

#         # end
#         h = self.norm_out(h)
#         h = nonlinearity(h)
#         h = self.conv_out(h)
#         return h


# class Encoder(nn.Module):
#     def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
#                  attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
#                  resolution, z_channels, double_z=True, **ignore_kwargs):
#         super().__init__()
#         self.ch = ch
#         self.temb_ch = 0
#         self.num_resolutions = len(ch_mult)
#         self.num_res_blocks = num_res_blocks
#         self.resolution = resolution
#         self.in_channels = in_channels

#         # downsampling
#         self.conv_in = torch.nn.Conv3d(in_channels,
#                                        self.ch,
#                                        kernel_size=3,
#                                        stride=1,
#                                        padding=1)

#         curr_res = resolution
#         in_ch_mult = (1,)+tuple(ch_mult)
#         self.down = nn.ModuleList()
#         for i_level in range(self.num_resolutions):
#             block = nn.ModuleList()
#             # attn = nn.ModuleList()
#             block_in = ch*in_ch_mult[i_level]
#             block_out = ch*ch_mult[i_level]
#             for i_block in range(self.num_res_blocks):
#                 block.append(ResnetBlock(in_channels=block_in,
#                                          out_channels=block_out,
#                                          temb_channels=self.temb_ch,
#                                          dropout=dropout))
#                 block_in = block_out
#                 # if curr_res in attn_resolutions:
#                 #     attn.append(FactorizedAttnBlock(block_in))
#             down = nn.Module()
#             down.block = block
#             # down.attn = attn
#             if i_level != self.num_resolutions-1:
#                 down.downsample = Downsample(block_in, resamp_with_conv)
#                 curr_res = curr_res // 2
#             self.down.append(down)

#         # middle
#         self.mid = nn.Module()
#         self.mid.block_1 = ResnetBlock(in_channels=block_in,
#                                        out_channels=block_in,
#                                        temb_channels=self.temb_ch,
#                                        dropout=dropout)
#         # self.mid.attn_1 = FactorizedAttnBlock(block_in)
#         self.mid.block_2 = ResnetBlock(in_channels=block_in,
#                                        out_channels=block_in,
#                                        temb_channels=self.temb_ch,
#                                        dropout=dropout)

#         # end
#         self.norm_out = Normalize(block_in)
#         self.conv_out = torch.nn.Conv3d(block_in,
#                                         2*z_channels if double_z else z_channels,
#                                         kernel_size=3,
#                                         stride=1,
#                                         padding=1)


#     def forward(self, x):
#         #assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

#         # timestep embedding
#         temb = None

#         # downsampling
#         hs = [self.conv_in(x)]
#         for i_level in range(self.num_resolutions):
#             for i_block in range(self.num_res_blocks):
#                 h = self.down[i_level].block[i_block](hs[-1], temb)
#                 # if len(self.down[i_level].attn) > 0:
#                 #     h = self.down[i_level].attn[i_block](h)
#                 hs.append(h)
#             if i_level != self.num_resolutions-1:
#                 hs.append(self.down[i_level].downsample(hs[-1]))

#         # middle
#         h = hs[-1]
#         h = self.mid.block_1(h, temb)
#         # h = self.mid.attn_1(h)
#         h = self.mid.block_2(h, temb)

#         # end
#         h = self.norm_out(h)
#         h = nonlinearity(h)
#         h = self.conv_out(h)
#         return h


# class Decoder(nn.Module):
#     def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
#                  attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
#                  resolution, z_channels, give_pre_end=False, **ignorekwargs):
#         super().__init__()
#         self.ch = ch
#         self.temb_ch = 0
#         self.num_resolutions = len(ch_mult)
#         self.num_res_blocks = num_res_blocks
#         self.resolution = resolution
#         self.in_channels = in_channels
#         self.give_pre_end = give_pre_end

#         # compute in_ch_mult, block_in and curr_res at lowest res
#         in_ch_mult = (1,)+tuple(ch_mult)
#         block_in = ch*ch_mult[self.num_resolutions-1]
#         curr_res = resolution // 2**(self.num_resolutions-1)
#         self.z_shape = (1,z_channels,curr_res,curr_res, curr_res)
#         print("Working with z of shape {} = {} dimensions.".format(
#             self.z_shape, np.prod(self.z_shape)))

#         # z to block_in
#         self.conv_in = torch.nn.Conv3d(z_channels,
#                                        block_in,
#                                        kernel_size=3,
#                                        stride=1,
#                                        padding=1)

#         # middle
#         self.mid = nn.Module()
#         self.mid.block_1 = ResnetBlock(in_channels=block_in,
#                                        out_channels=block_in,
#                                        temb_channels=self.temb_ch,
#                                        dropout=dropout)
#         # self.mid.attn_1 = FactorizedAttnBlock(block_in)
#         self.mid.block_2 = ResnetBlock(in_channels=block_in,
#                                        out_channels=block_in,
#                                        temb_channels=self.temb_ch,
#                                        dropout=dropout)

#         # upsampling
#         self.up = nn.ModuleList()
#         for i_level in reversed(range(self.num_resolutions)):
#             block = nn.ModuleList()
#             # attn = nn.ModuleList()
#             block_out = ch*ch_mult[i_level]
#             for i_block in range(self.num_res_blocks+1):
#                 block.append(ResnetBlock(in_channels=block_in,
#                                          out_channels=block_out,
#                                          temb_channels=self.temb_ch,
#                                          dropout=dropout))
#                 block_in = block_out
#                 # if curr_res in attn_resolutions:
#                 #     attn.append(FactorizedAttnBlock(block_in))
#             up = nn.Module()
#             up.block = block
#             # up.attn = attn
#             if i_level != 0:
#                 up.upsample = Upsample(block_in, resamp_with_conv)
#                 curr_res = curr_res * 2
#             self.up.insert(0, up) # prepend to get consistent order

#         # end
#         self.norm_out = Normalize(block_in)
#         self.conv_out = torch.nn.Conv3d(block_in,
#                                         out_ch,
#                                         kernel_size=3,
#                                         stride=1,
#                                         padding=1)

#     def forward(self, z):
#         #assert z.shape[1:] == self.z_shape[1:]
#         self.last_z_shape = z.shape

#         # timestep embedding
#         temb = None

#         # z to block_in
#         h = self.conv_in(z)

#         # middle
#         h = self.mid.block_1(h, temb)
#         # h = self.mid.attn_1(h)
#         h = self.mid.block_2(h, temb)

#         # upsampling
#         for i_level in reversed(range(self.num_resolutions)):
#             for i_block in range(self.num_res_blocks+1):
#                 h = self.up[i_level].block[i_block](h, temb)
#                 # if len(self.up[i_level].attn) > 0:
#                 #     h = self.up[i_level].attn[i_block](h)
#             if i_level != 0:
#                 h = self.up[i_level].upsample(h)

#         # end
#         if self.give_pre_end:
#             return h

#         h = self.norm_out(h)
#         h = nonlinearity(h)
#         h = self.conv_out(h)
#         return h


# class VUNet(nn.Module):
#     def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
#                  attn_resolutions, dropout=0.0, resamp_with_conv=True,
#                  in_channels, c_channels,
#                  resolution, z_channels, use_timestep=False, **ignore_kwargs):
#         super().__init__()
#         self.ch = ch
#         self.temb_ch = self.ch*4
#         self.num_resolutions = len(ch_mult)
#         self.num_res_blocks = num_res_blocks
#         self.resolution = resolution

#         self.use_timestep = use_timestep
#         if self.use_timestep:
#             # timestep embedding
#             self.temb = nn.Module()
#             self.temb.dense = nn.ModuleList([
#                 torch.nn.Linear(self.ch,
#                                 self.temb_ch),
#                 torch.nn.Linear(self.temb_ch,
#                                 self.temb_ch),
#             ])

#         # downsampling
#         self.conv_in = torch.nn.Conv3d(c_channels,
#                                        self.ch,
#                                        kernel_size=3,
#                                        stride=1,
#                                        padding=1)

#         curr_res = resolution
#         in_ch_mult = (1,)+tuple(ch_mult)
#         self.down = nn.ModuleList()
#         for i_level in range(self.num_resolutions):
#             block = nn.ModuleList()
#             attn = nn.ModuleList()
#             block_in = ch*in_ch_mult[i_level]
#             block_out = ch*ch_mult[i_level]
#             for i_block in range(self.num_res_blocks):
#                 block.append(ResnetBlock(in_channels=block_in,
#                                          out_channels=block_out,
#                                          temb_channels=self.temb_ch,
#                                          dropout=dropout))
#                 block_in = block_out
#                 # if curr_res in attn_resolutions:
#                 #     attn.append(FactorizedAttnBlock(block_in))
#             down = nn.Module()
#             down.block = block
#             down.attn = attn
#             if i_level != self.num_resolutions-1:
#                 down.downsample = Downsample(block_in, resamp_with_conv)
#                 curr_res = curr_res // 2
#             self.down.append(down)

#         self.z_in = torch.nn.Conv3d(z_channels,
#                                     block_in,
#                                     kernel_size=1,
#                                     stride=1,
#                                     padding=0)
#         # middle
#         self.mid = nn.Module()
#         self.mid.block_1 = ResnetBlock(in_channels=2*block_in,
#                                        out_channels=block_in,
#                                        temb_channels=self.temb_ch,
#                                        dropout=dropout)
#         # self.mid.attn_1 = FactorizedAttnBlock(block_in)
#         self.mid.block_2 = ResnetBlock(in_channels=block_in,
#                                        out_channels=block_in,
#                                        temb_channels=self.temb_ch,
#                                        dropout=dropout)

#         # upsampling
#         self.up = nn.ModuleList()
#         for i_level in reversed(range(self.num_resolutions)):
#             block = nn.ModuleList()
#             attn = nn.ModuleList()
#             block_out = ch*ch_mult[i_level]
#             skip_in = ch*ch_mult[i_level]
#             for i_block in range(self.num_res_blocks+1):
#                 if i_block == self.num_res_blocks:
#                     skip_in = ch*in_ch_mult[i_level]
#                 block.append(ResnetBlock(in_channels=block_in+skip_in,
#                                          out_channels=block_out,
#                                          temb_channels=self.temb_ch,
#                                          dropout=dropout))
#                 block_in = block_out
#                 # if curr_res in attn_resolutions:
#                 #     attn.append(FactorizedAttnBlock(block_in))
#             up = nn.Module()
#             up.block = block
#             up.attn = attn
#             if i_level != 0:
#                 up.upsample = Upsample(block_in, resamp_with_conv)
#                 curr_res = curr_res * 2
#             self.up.insert(0, up) # prepend to get consistent order

#         # end
#         self.norm_out = Normalize(block_in)
#         self.conv_out = torch.nn.Conv3d(block_in,
#                                         out_ch,
#                                         kernel_size=3,
#                                         stride=1,
#                                         padding=1)


#     def forward(self, x, z):
#         #assert x.shape[2] == x.shape[3] == self.resolution

#         if self.use_timestep:
#             # timestep embedding
#             assert t is not None
#             temb = get_timestep_embedding(t, self.ch)
#             temb = self.temb.dense[0](temb)
#             temb = nonlinearity(temb)
#             temb = self.temb.dense[1](temb)
#         else:
#             temb = None

#         # downsampling
#         hs = [self.conv_in(x)]
#         for i_level in range(self.num_resolutions):
#             for i_block in range(self.num_res_blocks):
#                 h = self.down[i_level].block[i_block](hs[-1], temb)
#                 if len(self.down[i_level].attn) > 0:
#                     h = self.down[i_level].attn[i_block](h)
#                 hs.append(h)
#             if i_level != self.num_resolutions-1:
#                 hs.append(self.down[i_level].downsample(hs[-1]))

#         # middle
#         h = hs[-1]
#         z = self.z_in(z)
#         h = torch.cat((h,z),dim=1)
#         h = self.mid.block_1(h, temb)
#         # h = self.mid.attn_1(h)
#         h = self.mid.block_2(h, temb)

#         # upsampling
#         for i_level in reversed(range(self.num_resolutions)):
#             for i_block in range(self.num_res_blocks+1):
#                 h = self.up[i_level].block[i_block](
#                     torch.cat([h, hs.pop()], dim=1), temb)
#                 if len(self.up[i_level].attn) > 0:
#                     h = self.up[i_level].attn[i_block](h)
#             if i_level != 0:
#                 h = self.up[i_level].upsample(h)

#         # end
#         h = self.norm_out(h)
#         h = nonlinearity(h)
#         h = self.conv_out(h)
#         return h


# class SimpleDecoder(nn.Module):
#     def __init__(self, in_channels, out_channels, *args, **kwargs):
#         super().__init__()
#         self.model = nn.ModuleList([nn.Conv3d(in_channels, in_channels, 1),
#                                      ResnetBlock(in_channels=in_channels,
#                                                  out_channels=2 * in_channels,
#                                                  temb_channels=0, dropout=0.0),
#                                      ResnetBlock(in_channels=2 * in_channels,
#                                                 out_channels=4 * in_channels,
#                                                 temb_channels=0, dropout=0.0),
#                                      ResnetBlock(in_channels=4 * in_channels,
#                                                 out_channels=2 * in_channels,
#                                                 temb_channels=0, dropout=0.0),
#                                      nn.Conv3d(2*in_channels, in_channels, 1),
#                                      Upsample(in_channels, with_conv=True)])
#         # end
#         self.norm_out = Normalize(in_channels)
#         self.conv_out = torch.nn.Conv3d(in_channels,
#                                         out_channels,
#                                         kernel_size=3,
#                                         stride=1,
#                                         padding=1)

#     def forward(self, x):
#         for i, layer in enumerate(self.model):
#             if i in [1,2,3]:
#                 x = layer(x, None)
#             else:
#                 x = layer(x)

#         h = self.norm_out(x)
#         h = nonlinearity(h)
#         x = self.conv_out(h)
#         return x


# class UpsampleDecoder(nn.Module):
#     def __init__(self, in_channels, out_channels, ch, num_res_blocks, resolution,
#                  ch_mult=(2,2), dropout=0.0):
#         super().__init__()
#         # upsampling
#         self.temb_ch = 0
#         self.num_resolutions = len(ch_mult)
#         self.num_res_blocks = num_res_blocks
#         block_in = in_channels
#         curr_res = resolution // 2 ** (self.num_resolutions - 1)
#         self.res_blocks = nn.ModuleList()
#         self.upsample_blocks = nn.ModuleList()
#         for i_level in range(self.num_resolutions):
#             res_block = []
#             block_out = ch * ch_mult[i_level]
#             for i_block in range(self.num_res_blocks + 1):
#                 res_block.append(ResnetBlock(in_channels=block_in,
#                                          out_channels=block_out,
#                                          temb_channels=self.temb_ch,
#                                          dropout=dropout))
#                 block_in = block_out
#             self.res_blocks.append(nn.ModuleList(res_block))
#             if i_level != self.num_resolutions - 1:
#                 self.upsample_blocks.append(Upsample(block_in, True))
#                 curr_res = curr_res * 2

#         # end
#         self.norm_out = Normalize(block_in)
#         self.conv_out = torch.nn.Conv3d(block_in,
#                                         out_channels,
#                                         kernel_size=3,
#                                         stride=1,
#                                         padding=1)

#     def forward(self, x):
#         # upsampling
#         h = x
#         for k, i_level in enumerate(range(self.num_resolutions)):
#             for i_block in range(self.num_res_blocks + 1):
#                 h = self.res_blocks[i_level][i_block](h, None)
#             if i_level != self.num_resolutions - 1:
#                 h = self.upsample_blocks[k](h)
#         h = self.norm_out(h)
#         h = nonlinearity(h)
#         h = self.conv_out(h)
#         return h

