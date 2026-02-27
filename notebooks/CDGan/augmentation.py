"""
Data Augmentation using trained GAN
Use this to augment your CNN training datasets with synthetic X-ray images
"""
import torch
from torchvision.utils import save_image
import os
from notebooks import Generator, ConditionalGenerator


class GANAugmenter:
    """Use trained GAN to augment training data for CNN models"""
    
    def __init__(self, generator_path, conditional=False, num_classes=3, 
                 latent_dim=100, img_size=256, device=None):
        """
        Initialize GAN augmenter
        
        Args:
            generator_path: Path to trained generator weights
            conditional: Whether using conditional GAN
            num_classes: Number of classes (for conditional GAN)
            latent_dim: Latent dimension size
            img_size: Image size
            device: Device to run on
        """
        self.conditional = conditional
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load generator
        if conditional:
            self.generator = ConditionalGenerator(
                latent_dim=latent_dim,
                num_classes=num_classes,
                img_size=img_size
            ).to(self.device)
        else:
            self.generator = Generator(
                latent_dim=latent_dim,
                img_size=img_size
            ).to(self.device)
        
        self.generator.load_state_dict(torch.load(generator_path, map_location=self.device))
        self.generator.eval()
        print(f"Generator loaded from {generator_path}")
    
    def generate_batch(self, batch_size, class_label=None):
        """
        Generate a batch of synthetic images
        
        Args:
            batch_size: Number of images to generate
            class_label: Class label for conditional generation (0=empty, 1=overlap, 2=clutter)
        
        Returns:
            Tensor of generated images (batch_size, 1, img_size, img_size)
        """
        with torch.no_grad():
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
            
            if self.conditional:
                if class_label is None:
                    # Generate random labels if not specified
                    labels = torch.randint(0, self.num_classes, (batch_size,)).to(self.device)
                else:
                    labels = torch.full((batch_size,), class_label).to(self.device)
                
                images = self.generator(z, labels)
                return images, labels
            else:
                images = self.generator(z)
                return images
    
    def augment_dataset(self, output_dir, num_images_per_class, class_names=None):
        """
        Generate synthetic images for each class to augment dataset
        
        Args:
            output_dir: Directory to save generated images
            num_images_per_class: Number of images to generate per class
            class_names: List of class names (e.g., ['empty', 'overlap', 'clutter'])
        """
        if class_names is None:
            class_names = [f"class_{i}" for i in range(self.num_classes)]
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.conditional:
            for class_idx, class_name in enumerate(class_names):
                class_dir = os.path.join(output_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                print(f"Generating {num_images_per_class} images for class: {class_name}")
                
                for i in range(num_images_per_class):
                    images, _ = self.generate_batch(1, class_label=class_idx)
                    save_image(
                        images,
                        os.path.join(class_dir, f"synthetic_{i+1}.png"),
                        normalize=True
                    )
                
                print(f"Saved {num_images_per_class} images to {class_dir}")
        else:
            print(f"Generating {num_images_per_class} synthetic images...")
            for i in range(num_images_per_class):
                images = self.generate_batch(1)
                save_image(
                    images,
                    os.path.join(output_dir, f"synthetic_{i+1}.png"),
                    normalize=True
                )
            print(f"Saved {num_images_per_class} images to {output_dir}")
    
    def online_augmentation(self, real_images, augmentation_ratio=0.5):
        """
        Perform online augmentation during training
        Mix real images with synthetic ones in each batch
        
        Args:
            real_images: Batch of real images
            augmentation_ratio: Ratio of synthetic to real images (0.5 = 50% synthetic)
        
        Returns:
            Augmented batch combining real and synthetic images
        """
        batch_size = real_images.size(0)
        num_synthetic = int(batch_size * augmentation_ratio)
        
        if num_synthetic > 0:
            synthetic_images = self.generate_batch(num_synthetic)
            if self.conditional:
                synthetic_images = synthetic_images[0]  # Extract images from tuple
            
            # Combine real and synthetic
            augmented_batch = torch.cat([real_images, synthetic_images], dim=0)
            
            # Shuffle
            indices = torch.randperm(augmented_batch.size(0))
            augmented_batch = augmented_batch[indices]
            
            return augmented_batch
        
        return real_images


def balance_dataset_with_gan(generator_path, dataset_dir, target_count_per_class, 
                             conditional=True, class_names=None):
    """
    Balance an imbalanced dataset by generating synthetic samples for underrepresented classes
    
    Args:
        generator_path: Path to trained generator
        dataset_dir: Root directory containing class subdirectories
        target_count_per_class: Target number of images per class
        conditional: Whether using conditional GAN
        class_names: List of class names
    """
    if class_names is None:
        class_names = ['empty', 'overlap', 'clutter']
    
    augmenter = GANAugmenter(
        generator_path=generator_path,
        conditional=conditional,
        num_classes=len(class_names)
    )
    
    # Count existing images per class
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            current_count = 0
        else:
            current_count = len([f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        needed = target_count_per_class - current_count
        
        if needed > 0:
            print(f"\nClass '{class_name}': {current_count} images, generating {needed} more...")
            
            for i in range(needed):
                images, _ = augmenter.generate_batch(1, class_label=class_idx)
                save_image(
                    images,
                    os.path.join(class_dir, f"synthetic_{current_count + i + 1}.png"),
                    normalize=True
                )
            
            print(f"Class '{class_name}' now has {target_count_per_class} images")
        else:
            print(f"Class '{class_name}': {current_count} images (already sufficient)")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Augment dataset with GAN")
    parser.add_argument("--generator_path", type=str, required=True, 
                       help="Path to trained generator")
    parser.add_argument("--output_dir", type=str, default="./augmented_data",
                       help="Output directory for synthetic images")
    parser.add_argument("--num_images_per_class", type=int, default=500,
                       help="Number of synthetic images to generate per class")
    parser.add_argument("--conditional", action="store_true",
                       help="Use conditional GAN")
    parser.add_argument("--balance_dataset", action="store_true",
                       help="Balance existing dataset instead of creating new one")
    parser.add_argument("--dataset_dir", type=str, default="./data/train",
                       help="Dataset directory (for balancing)")
    parser.add_argument("--target_count", type=int, default=1000,
                       help="Target number of images per class (for balancing)")
    
    args = parser.parse_args()
    
    class_names = ['empty', 'overlap', 'clutter']
    
    if args.balance_dataset:
        # Balance existing dataset
        balance_dataset_with_gan(
            generator_path=args.generator_path,
            dataset_dir=args.dataset_dir,
            target_count_per_class=args.target_count,
            conditional=args.conditional,
            class_names=class_names
        )
    else:
        # Generate new synthetic dataset
        augmenter = GANAugmenter(
            generator_path=args.generator_path,
            conditional=args.conditional,
            num_classes=len(class_names)
        )
        
        augmenter.augment_dataset(
            output_dir=args.output_dir,
            num_images_per_class=args.num_images_per_class,
            class_names=class_names
        )