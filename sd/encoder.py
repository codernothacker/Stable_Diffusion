import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResdiualBlock

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            #(BAtch_Size, channel , height, width) -> (Batch_size, 128, height, width)
        nn.Conv2d(3,128,kernel_size=3, padding=1),

        #(batch_size , 128. heigth , width) -> (batch_size , 128, height, width)
        VAE_ResdiualBlock(128,128),

        #(batch_size, 128, height, width) -> (batch_size, 128, height/2, width/2)
        nn.Conv2d(128, 128, kernel_size=3, stride=2,padding=0),

        #(batch_size , 128, height/2, width/2) -> (batch_size , 256 ,height/2 , width/2)
        VAE_ResdiualBlock(128,256),

        #(batch_size, 256, height/2 , width/2) -> (batch_size , 256, heigth/2 ,width/2)
        VAE_ResdiualBlock(256, 256),

        #(batch_size, 256, height/2 , width/2) -> (batch_size , 256, heigth/2 ,width/2)
        nn.Conv2d(256, 256, kernel_size=3, stride=2,padding=0),

        #(batch_size, 256, height/4 , width/4) -> (batch_size , 256, heigth/4 ,width/4)
        VAE_ResdiualBlock(256, 512),

        #(batch_size, 512, height/4 , width/4) -> (batch_size , 512, heigth/4 ,width/4)
        VAE_ResdiualBlock(512, 512),

        #(batch_size, 512, height/4 , width/4) -> (batch_size , 512, heigth/8 ,width/8)
        nn.Conv2d(512, 512, kernel_size=3, stride=2,padding=0),
        
        VAE_ResdiualBlock(512, 512),

        VAE_ResdiualBlock(512, 512),

        #(batch_size, 512, height/8 , width/8) -> (batch_size , 512, heigth/8 ,width/8)
        VAE_ResdiualBlock(512, 512),

        #(batch_size, 512, height/8 , width/8) -> (batch_size , 512, heigth/8 ,width/8)
        VAE_AttentionBlock(512),

        #(batch_size, 512, height/8 , width/8) -> (batch_size , 512, heigth/8 ,width/8)
        VAE_ResdiualBlock(512, 512),

        #(batch_size, 512, height/8 , width/8) -> (batch_size , 512, heigth/8 ,width/8)
        nn.GroupNorm(32, 512),

        #(batch_size, 512, height/8 , width/8) -> (batch_size , 512, heigth/8 ,width/8)
        nn.SiLU(),

        #(batch_size, 512, height/8 , width/8) -> (batch_size , 8, heigth/8 ,width/8)
        nn.Conv2d(512, 8, kernel_size=3, padding =1),

        #(batch_size, 8, height/8 , width/8) -> (batch_size , 8, heigth/8 ,width/8)
        nn.Conv2d(8, 8, kernel_size=1, padding =0)

        )

def forward(self, x:torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    # x: (batch_size , channel. height , width)
    # noise : (batch_size , out_channel ,height/8, width/8)

    for module in self:
        if getattr(module, 'stride' ,None) ==(2,2):
            #(padding_left, padding_right, padding_top, padding_bottom)
            x=F.pad(x,(0,1,0,1))
        x=module(x)
    #(batch_size, 8, height/8, width/8) -> two tensor of shape (batch_size , 4, height/8, width/8)
    mean , log_variance = torch.chunk(x,2, dim=1)

    log_variance= torch.clamp(log_variance, -30 ,20)
    
    #(batch_size, 8, height/8, width/8) -> two tensor of shape (batch_size , 4, height/8, width/8)
    variance= log_variance.exp()

    #(batch_size, 8, height/8, width/8) -> two tensor of shape (batch_size , 4, height/8, width/8)
    stdev= variance.sqrt()

    #N(0,1) -> N(mean, variance)=x?
    #X= mean  + stdev *2
    x= mean + stdev * noise

    #scal the output by a constant
    x *= 0.18215
    return x 
