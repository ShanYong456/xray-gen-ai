import torch.nn as nn

# Weight Initialization

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)


# Generator: z -> 1 x 64 x 64

class XRayGenerator(nn.Module):
     """
    DCGAN-style generator for grayscale 64x64 images.
    Input: latent vector z of shape [B, nz, 1, 1]
    Output: [B, 1, 64, 64] with values in [-1, 1] (tanh)
    """
     
     def __init__(self, nz=100, ngf=64, nc=1):
        super().__init__()
        self.main = nn.Sequential(
             # input Z goes into a convolution

             nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), # 4x4
             nn.BatchNorm2d(ngf * 8),
             nn.ReLU(True),

             # state size: (ngf*8) x 4 x 4
             nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), # 8x8
             nn.BatchNorm2d(ngf * 4),
             nn.ReLU(True),

             # state size: (ngf*4) x 8 x 8
             nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), # 16x16
             nn.BatchNorm2d(ngf * 2),
             nn.ReLU(True),

             # state size: (ngf*2) x 16 x 16
             nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), # 32x32
             nn.BatchNorm2d(ngf),
             nn.ReLU(True),

             # state size: (ngf) x 32 x 32
             nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), # 64x64
             nn.Tanh(), # output in [-1, 1]

        )
    
     def forward(self, z):
        return self.main(z)
     

# Discriminator: 1 x 64 x 64 -> real/fake

class XRayDiscriminator(nn.Module):
    """
    DCGAN-style discriminator for grayscale 64x64 images.
    Input: [B, 1, 64, 64] with values in [-1, 1]
    Output: [B, 1] logits (no sigmoid inside)
    """

    def __init__(self, ndf=64, nc=1):
        super().__init__()
        self.main = nn.Sequential(
            #input: (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), # 32x32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), # 16x16
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), # 8x8
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf *4, ndf * 8, 4, 2, 1, bias=False), # 4x4
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), # 1x1

        )

    def forward(self, x):
        out = self.main(x) # [8, 1, 1, 1]
        return out.view(-1, 1)
    

