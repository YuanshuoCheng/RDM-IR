import torch
import torch.nn as nn
from Restoration.nets import DoubleConv,DownBlock,UpBlock,Conv3x3,OutConv,SingleConv
from Restoration.attention import CrossAttention,MultiScaleAttention,LayerNorm,FeedForward



class UpdateQ(nn.Module):
    def __init__(self):
        super(UpdateQ, self).__init__()
    def forward(self,P,B,x,y,rou):
        res = (rou*B+P*x-y)/(rou+1)
        return res

class Encoder(nn.Module):
    def __init__(self,in_channels):
        super(Encoder, self).__init__()
        base_dim = in_channels
        self.encoderBlock0 = DownBlock(base_dim, base_dim * 2)
        self.encoderBlock1 = DownBlock(base_dim * 2, base_dim * 4)
        self.encoderBlock2 = DownBlock(base_dim * 4, base_dim * 8)
        self.encoderBlock3 = DownBlock(base_dim * 8, base_dim * 8)
    def forward(self,x):
        x,feat0 = self.encoderBlock0(x)
        x, feat1 = self.encoderBlock1(x)
        x, feat2 = self.encoderBlock2(x)
        x, feat3 = self.encoderBlock3(x)
        return x,feat0,feat1,feat2,feat3

class Decoder(nn.Module):
    def __init__(self,out_channels):
        super(Decoder, self).__init__()
        base_dim = out_channels
        self.decoderBlock0 = UpBlock(base_dim*16,base_dim*4)
        self.decoderBlock1 = UpBlock(base_dim*8,base_dim*2)
        self.decoderBlock2 = UpBlock(base_dim*4,base_dim)
        self.decoderBlock3 = UpBlock(base_dim*2,base_dim)
    def forward(self,x,feat0,feat1,feat2,feat3):
        x = self.decoderBlock0(x,feat3)
        x = self.decoderBlock1(x, feat2)
        x = self.decoderBlock2(x, feat1)
        x = self.decoderBlock3(x, feat0)
        return x


class UNET(nn.Module):
    def __init__(self,in_channels,out_channels,base_dim=32):
        super(UNET, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels!= self.out_channels:
            self.deg_c = out_channels//2
        self.head = DoubleConv(in_channels,base_dim,mid_channels=base_dim)
        self.tail = DoubleConv(base_dim,base_dim,mid_channels=base_dim)
        self.out = OutConv(base_dim,out_channels)
        #self.n_blocks = config.n_blocks
        self.encoderBlock0 = DownBlock(base_dim,base_dim*2)
        self.encoderBlock1 = DownBlock(base_dim*2,base_dim*4)
        self.encoderBlock2 = DownBlock(base_dim*4,base_dim*8)
        self.encoderBlock3 = DownBlock(base_dim*8,base_dim*8)

        self.decoderBlock0 = UpBlock(base_dim*16,base_dim*4)
        self.decoderBlock1 = UpBlock(base_dim*8,base_dim*2)
        self.decoderBlock2 = UpBlock(base_dim*4,base_dim)
        self.decoderBlock3 = UpBlock(base_dim*2,base_dim)
    def forward(self,img):
        x = self.head(img)
        x,feat0 = self.encoderBlock0(x)
        x, feat1 = self.encoderBlock1(x)
        x, feat2 = self.encoderBlock2(x)
        x, feat3 = self.encoderBlock3(x)

        x = self.decoderBlock0(x,feat3)
        x = self.decoderBlock1(x, feat2)
        x = self.decoderBlock2(x, feat1)
        x = self.decoderBlock3(x, feat0)
        x = self.tail(x)
        x = self.out(x)
        if self.in_channels!= self.out_channels:
            Arec,B = x[:,:self.deg_c,::],x[:,self.deg_c:,::]
            return (img-B)*Arec,Arec,B
        else:
            return x

class MultiScaleChannelTransformer(nn.Module):
    def __init__(self, in_channels=3,out_channels=3,mid_channels=128,ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(MultiScaleChannelTransformer, self).__init__()
        dim = mid_channels//2
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = MultiScaleAttention(dim, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim=dim,ffn_expansion_factor=ffn_expansion_factor, bias=bias)

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels//2, kernel_size=3,padding=1,stride=1),
            nn.GroupNorm(4,mid_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels//2, mid_channels // 2, kernel_size=3, padding=1, stride=1),
            nn.GroupNorm(4, mid_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels//2, mid_channels//2, kernel_size=3, padding=1, stride=1)
        )
        self.proj_out = nn.Sequential(
            nn.Conv2d(mid_channels//2, mid_channels//2, kernel_size=3,padding=1,stride=1),
            nn.GroupNorm(4,mid_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels//2, out_channels, kernel_size=3, padding=1, stride=1)
        )

    def forward(self, x):
        x = self.proj_in(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return self.proj_out(x)

class PriorMixer(nn.Module):
    def __init__(self,in_channels=6,out_channels=6,base_dim=32):
        super(PriorMixer, self).__init__()
        self.deg_c = out_channels // 2
        self.head = DoubleConv(in_channels, base_dim, mid_channels=base_dim)
        self.tail = DoubleConv(base_dim, base_dim,mid_channels=base_dim)
        self.encoder = Encoder(in_channels=base_dim)
        self.decoder = Decoder(out_channels=base_dim)
        self.out = OutConv(base_dim, out_channels)
        self.proj = SingleConv(in_channels=base_dim*8,out_channels=base_dim*8,strid=2)
        self.cross_attention = CrossAttention(dim=base_dim*8)
        self.mixer = nn.Conv2d(in_channels=base_dim*16,out_channels=base_dim*8,kernel_size=3,stride=1,padding=1)


    def forward(self,ArecB,PQ):
        ArecB = self.head(ArecB)
        PQ = self.head(PQ)
        x, feat0, feat1, feat2, feat3 = self.encoder(ArecB)
        ref,_,_,_,_ = self.encoder(PQ)
        feat_ref = self.proj(ref)
        feat_ref = self.cross_attention(x,feat_ref)
        x = self.mixer(torch.cat([x,feat_ref],dim=1))
        x = self.decoder(x,feat0,feat1,feat2,feat3)
        x = self.tail(x)
        ArecB = self.out(x)
        Arec, B = ArecB[:, :self.deg_c, ::], ArecB[:, self.deg_c:, ::]
        return Arec, B
