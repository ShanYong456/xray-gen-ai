import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as T

from xraygen.data.dataset import XRayDataset
from xraygen.models.generator import (
    XRayGenerator,
    XRayDiscriminator,
    weights_init_normal,
)

def train_gan(
        csv_path="data/raw/metadata_sample.csv",
        img_root="data/raw",
        nz=100,
        batch_size=32,
        num_epochs=10,
        lr=2e-4,
        beta1=0.5,
        out_dir="models/generator",
        device=None,
):
    
    # 1. Device

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GAN] Using device: {device}")


    # 2. Dataset & DataLoader

    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),                 # [1, H, W] in [0,1]
        T.Normalize((0.5,), (0.5,)),  # -> [-1, 1] for DCGAN
    ])

    dataset = XRayDataset(csv_path, img_root, transform=transform)
    if len(dataset) == 0:
        raise ValueError("[GAN] Empty dataset! check CSV and image paths.")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"[GAN] Dataset size: {len(dataset)}, batches/epoch: {len(dataloader)}")

    # 3. Models

    netG = XRayGenerator(nz=nz).to(device)
    netD = XRayDiscriminator().to(device)

    netG.apply(weights_init_normal)
    netD.apply(weights_init_normal)

    print("[GAN] Generator architecture:\n", netG)
    print("[GAN] Discriminator architecture:\n", netD)


    # 4. Loss & Optimizers

    criterion = nn.BCEWithLogitsLoss()

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    fixed_noise = torch.rand(16, nz, 1, 1, device=device) # for monitoring


    # 5. Training Loop

    for epoch in range(1, num_epochs + 1):
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            bsz = real_imgs.size(0)

            # Labels
            real_labels = torch.ones(bsz, 1, device=device)
            fake_labels = torch.zeros(bsz, 1, device=device)

            # 1. Update Discriminator
            netD.zero_grad()

            # Real
            out_real = netD(real_imgs)
            lossD_real = criterion(out_real, real_labels)

            # Fake
            z = torch.randn(bsz, nz, 1, 1, device=device)
            fake_imgs = netG(z).detach() # detach so G not updated here
            out_fake = netD(fake_imgs)
            lossD_fake = criterion(out_fake, fake_labels)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()

            # 2. Update Generator
            netG.zero_grad()

            z = torch.randn(bsz, nz, 1, 1, device=device)
            fake_imgs = netG(z)
            out_fake_for_G = netD(fake_imgs)
            lossG = criterion(out_fake_for_G, real_labels) # want fake to be "real"
            
            lossG.backward()
            optimizerG.step()

            if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
                print(
                    f"[Epoch {epoch}/{num_epochs}] "
                    f"[Batch {i+1}/{len(dataloader)}] "
                    f"LossD: {lossD.item():.4f} | LossG: {lossG.item():.4f}"
                )
        
        # Optionally: generate a fixed batch for visual monitoring
        with torch.no_grad():
            fake_fixed = netG(fixed_noise).cpu()
            #  can save this as image using torchvision.utils.save_image
        
    # 6. Save Generator
    os.makedirs(out_dir, exist_ok=True)
    gen_path = os.path.join(out_dir, "generator.pt")
    torch.save(netG.state_dict(), gen_path)
    print("f[GAN] Saved generator weight to: {gen_path}")

def main():
    # Later we can load from configs; for now, use defaults.
    train_gan()

if __name__== "__main__":
    main()