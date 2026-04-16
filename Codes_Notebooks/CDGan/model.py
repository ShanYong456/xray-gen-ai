"""
GAN Models for Synthetic X-ray Image Generation
Generator and Discriminator for creating realistic X-ray tray images
"""
import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Generator network to create synthetic X-ray images from random noise
    """
    def __init__(self, latent_dim=100, img_channels=1, img_size=256):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size
        
        # Initial dense layer
        self.init_size = img_size // 16  # 16 for 4 upsampling layers
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * self.init_size * self.init_size),
            nn.BatchNorm1d(512 * self.init_size * self.init_size),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling layers
        self.conv_blocks = nn.Sequential(
            # 16x16 -> 32x32
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 128x128 -> 256x256
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Final layer to get image
            nn.Conv2d(32, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output range [-1, 1]
        )
    
    def forward(self, z):
        """
        Forward pass
        Args:
            z: Random noise vector (batch_size, latent_dim)
        Returns:
            Generated images (batch_size, img_channels, img_size, img_size)
        """
        x = self.fc(z)
        x = x.view(x.size(0), 512, self.init_size, self.init_size)
        img = self.conv_blocks(x)
        return img


class Discriminator(nn.Module):
    """
    Discriminator network to classify real vs fake X-ray images
    """
    def __init__(self, img_channels=1, img_size=256):
        super(Discriminator, self).__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        
        def discriminator_block(in_channels, out_channels, normalize=True):
            """Helper function to create discriminator blocks"""
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            # 256x256 -> 128x128
            *discriminator_block(img_channels, 32, normalize=False),
            
            # 128x128 -> 64x64
            *discriminator_block(32, 64),
            
            # 64x64 -> 32x32
            *discriminator_block(64, 128),
            
            # 32x32 -> 16x16
            *discriminator_block(128, 256),
            
            # 16x16 -> 8x8
            *discriminator_block(256, 512),
        )
        
        # Calculate size after conv layers
        ds_size = img_size // (2 ** 5)  # 5 downsampling layers
        
        # Final classification layer
        self.adv_layer = nn.Sequential(
            nn.Linear(512 * ds_size * ds_size, 1),
            #nn.Sigmoid()
        )
    
    def forward(self, img):
        """
        Forward pass
        Args:
            img: Input images (batch_size, img_channels, img_size, img_size)
        Returns:
            Probability that image is real (batch_size, 1)
        """
        features = self.model(img)
        features = features.view(features.size(0), -1)
        validity = self.adv_layer(features)
        return validity


# Conditional GAN (optional - for controlled generation)
class ConditionalGenerator(nn.Module):
    """
    Conditional Generator - can generate specific types of X-ray images
    (e.g., empty tray, overlapped items, cluttered items)
    """
    def __init__(self, latent_dim=100, num_classes=3, img_channels=1, img_size=256):
        super(ConditionalGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        
        self.init_size = img_size // 16
        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2, 512 * self.init_size * self.init_size),
            nn.BatchNorm1d(512 * self.init_size * self.init_size),
            nn.ReLU(inplace=True)
        )
        
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        """
        Forward pass with class conditioning
        Args:
            z: Random noise (batch_size, latent_dim)
            labels: Class labels (batch_size,) - 0: empty, 1: overlap, 2: clutter
        Returns:
            Generated images
        """
        label_input = self.label_embedding(labels)
        x = torch.cat([z, label_input], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 512, self.init_size, self.init_size)
        img = self.conv_blocks(x)
        return img


class ConditionalDiscriminator(nn.Module):
    """
    Conditional Discriminator - evaluates images with class information
    """
    def __init__(self, num_classes=3, img_channels=1, img_size=256):
        super(ConditionalDiscriminator, self).__init__()
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size
        
        # Label embedding projected to image space
        self.label_embedding = nn.Embedding(num_classes, img_size * img_size)
        
        def discriminator_block(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        # Process image + label
        self.model = nn.Sequential(
            *discriminator_block(img_channels + 1, 32, normalize=False),  # +1 for label channel
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )
        
        ds_size = img_size // (2 ** 5)
        self.adv_layer = nn.Sequential(
            nn.Linear(512 * ds_size * ds_size, 1),
            #nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        """
        Forward pass with conditioning
        Args:
            img: Input images
            labels: Class labels
        Returns:
            Probability that image is real
        """
        # Create label map
        label_input = self.label_embedding(labels)
        label_input = label_input.view(labels.size(0), 1, self.img_size, self.img_size)
        
        # Concatenate image and label
        d_in = torch.cat([img, label_input], dim=1)
        
        features = self.model(d_in)
        features = features.view(features.size(0), -1)
        validity = self.adv_layer(features)
        return validity


if __name__ == "__main__":
    # Test the models
    print("Testing GAN models...")
    
    # Test standard GAN
    latent_dim = 100
    batch_size = 4
    img_size = 256
    
    generator = Generator(latent_dim=latent_dim, img_size=img_size)
    discriminator = Discriminator(img_size=img_size)
    
    z = torch.randn(batch_size, latent_dim)
    fake_images = generator(z)
    print(f"Generated images shape: {fake_images.shape}")
    
    validity = discriminator(fake_images)
    print(f"Discriminator output shape: {validity.shape}")
    
    # Test conditional GAN
    print("\nTesting Conditional GAN...")
    cond_generator = ConditionalGenerator(latent_dim=latent_dim, num_classes=3, img_size=img_size)
    cond_discriminator = ConditionalDiscriminator(num_classes=3, img_size=img_size)
    
    labels = torch.randint(0, 3, (batch_size,))
    fake_images = cond_generator(z, labels)
    print(f"Conditional generated images shape: {fake_images.shape}")
    
    validity = cond_discriminator(fake_images, labels)
    print(f"Conditional discriminator output shape: {validity.shape}")
    
    print("\nAll models initialized successfully!")