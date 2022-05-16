from utils import utilities
from torch import nn
import pytorch_lightning as pl

def assemble_decoder_input():
    # Learn decoder input
    return nn.Sequential(
        nn.Linear(64, 128,bias=False),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 256,bias=False),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 512,bias=False),
        nn.BatchNorm1d(512),
    )
    
def assemble_decoder():
    # Assemble decoder 
    return nn.Sequential(
        DeConv(128,64,4), # 4x4
        DeConv(64,64,4), # 8x8
        DeConv(64,64,4), # 16x16
        DeConv(64,32,4), # 32x32
        DeConv(32,16,4), # 64x64
        DeConv(16,1,4,batch_normalisation=False, activation=False) # 128x128
    )

class DeConv(pl.LightningModule):

    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size,
                 batch_normalisation=True,
                 output_padding=0,
                 activation=True
                ):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, padding=1, stride=2, output_padding=output_padding, bias=(not batch_normalisation)),
        ]

        if batch_normalisation:
            layers.append(nn.BatchNorm2d(out_ch))
        if activation:
            layers.append(nn.ReLU())
        self.trans_conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.trans_conv(x)
        return x
    
class Decoder(pl.LightningModule):
    
    """
    Decoder base model
    """
    def __init__(self):
        super().__init__()
        
        # Learn decoder input
        self.decoder_input = assemble_decoder_input()
        
        # Decoder backbone
        self.decoder = assemble_decoder()
    
class ImageDecoder(Decoder):
    """
    Decoder for image models
    """
    def __init__(self, pretrain=False):
        super().__init__()
        if pretrain:
            utilities.load_pretrain(self)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.decoder_input(x)
        x = x.reshape(B,128,2,2)
        x = self.decoder(x)
        H, W = x.shape[-2:]
        x = x.reshape(B, 1, H, W)
        return x

# ImageDecoder wrapper for variational model (important for pretraining pipeline)
class VariationalImageDecoder(ImageDecoder):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
    
class VideoDecoder(Decoder):
    """
    Decoder for video models
    """
    def __init__(self, pretrain=False, trajectory_func='orig'):
        super().__init__()
        
        # Save Trajectory function for forward call        
        self.trajectory_func = trajectory_func
        
        if pretrain:
            utilities.load_pretrain(self)

    def forward(self, x):
        B, T = x.shape[:2]
        x = x.reshape(B*T, -1)
        x = self.decoder_input(x)
        x = x.reshape(B*T,128,2,2)
        x = self.decoder(x)
        H, W = x.shape[-2:]
        x = x.reshape(B, T, 1, H, W)
        return x

# VideoDecoder wrapper for variational model (important for pretraining pipeline)
class VariationalVideoDecoder(VideoDecoder):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    