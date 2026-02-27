"""
Dataset loader for X-ray images
This file loads your X-ray images and prepares them for GAN training
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


class XrayDataset(Dataset):
    
    
    def __init__(self, data_root, img_size=256, conditional=False, transform=None):
        """
        Initialize dataset
        
        Args:
            data_root: Root directory containing images (default: ./data/xray_images)
            img_size: Size to resize images to (default: 256x256)
            conditional: If True, expects class subdirectories (empty/overlap/clutter)
            transform: Optional custom transforms (if None, uses default)
        """
        self.data_root = data_root
        self.img_size = img_size
        self.conditional = conditional
        
        # Default transforms for X-ray images
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),  # Resize to square
                transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
                transforms.ToTensor(),  # Convert to tensor [0, 1]
                transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
            ])
        else:
            self.transform = transform
        
        self.images = []
        self.labels = []
        
        if conditional:
            # Load images from class subdirectories
            self.class_names = ['empty', 'overlap', 'clutter']
            self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            
            print(f"Loading conditional dataset from: {data_root}")
            
            for class_name in self.class_names:
                class_dir = os.path.join(data_root, class_name)
                
                if not os.path.exists(class_dir):
                    print(f"Warning: Directory '{class_dir}' not found, skipping...")
                    continue
                
                class_idx = self.class_to_idx[class_name]
                
                # Get all image files in this class directory
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                        img_path = os.path.join(class_dir, img_name)
                        self.images.append(img_path)
                        self.labels.append(class_idx)
            
            # Print statistics
            print(f"\nLoaded {len(self.images)} images across {len(self.class_names)} classes:")
            for class_name in self.class_names:
                count = self.labels.count(self.class_to_idx[class_name])
                print(f"   - {class_name}: {count} images")
            print()
            
        else:
            # Load all images from root directory (non-conditional)
            print(f"Loading dataset from: {data_root}")
            
            if not os.path.exists(data_root):
                raise ValueError(f"Data root directory '{data_root}' does not exist!")
            
            for img_name in os.listdir(data_root):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                    img_path = os.path.join(data_root, img_name)
                    self.images.append(img_path)
            
            print(f"Loaded {len(self.images)} images\n")
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {data_root}! Check your directory structure.")
    
    def __len__(self):
        """Return total number of images"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get a single image (and label if conditional)
        
        Args:
            idx: Index of image to get
            
        Returns:
            If conditional: (image_tensor, label)
            If not conditional: image_tensor
        """
        img_path = self.images[idx]
        
        try:
            # Load image and convert to grayscale
            image = Image.open(img_path).convert('L')  # 'L' mode = grayscale
            
            # Apply transforms
            image = self.transform(image)
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = torch.zeros(1, self.img_size, self.img_size)
        
        if self.conditional:
            label = self.labels[idx]
            return image, label
        else:
            return image


def Get_dataloader(data_root, batch_size=32, img_size=256, conditional=False, 
                   num_workers=4, shuffle=True):
    """
    Create a DataLoader for X-ray images
    
    Args:
        data_root: Root directory containing images (default: ./data/xray_images)
        batch_size: Number of images per batch
        img_size: Size to resize images to
        conditional: Whether to use conditional GAN (requires class subdirectories)
        num_workers: Number of parallel workers for loading data
        shuffle: Whether to shuffle the data
    
    Returns:
        DataLoader object ready for training
    """
    
    # Create dataset
    dataset = XrayDataset(
        data_root=data_root,
        img_size=img_size,
        conditional=conditional
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # Drop last incomplete batch
    )
    
    return dataloader


# Test and example usage
if __name__ == "__main__":
    """
    Test the dataset loader
    
    Usage:
        python utils/dataset.py --data_root ./my_xray_data --conditional
        python utils/dataset.py --data_root ./my_xray_data --batch_size 8
    """
    import argparse

    data_root="./data/Stage0/Color"
    
    parser = argparse.ArgumentParser(description="Test X-ray dataset loader")
    parser.add_argument("--data_root", type=str, required=True, 
                       help="Path to data directory")
    parser.add_argument("--conditional", action="store_true", 
                       help="Use conditional dataset (expects class subdirectories)")
    parser.add_argument("--batch_size", type=int, default=4, 
                       help="Batch size for testing")
    parser.add_argument("--img_size", type=int, default=256,
                       help="Image size")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Testing X-ray Dataset Loader")
    print("=" * 60)
    print(f"Data root: {args.data_root}")
    print(f"Conditional: {args.conditional}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}")
    print("=" * 60)
    print()
    
    try:
        # Create dataloader
        dataloader = Get_dataloader(
            data_root=args.data_root,
            batch_size=args.batch_size,
            img_size=args.img_size,
            conditional=args.conditional,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            shuffle=True
        )
        
        print(f"Total batches in dataset: {len(dataloader)}")
        print(f"Total images: {len(dataloader.dataset)}")
        print()
        
        # Load and display info about first batch
        print("Loading first batch...")
        
        if args.conditional:
            images, labels = next(iter(dataloader))
            print(f"Images shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Labels in batch: {labels.tolist()}")
            print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"Expected range: [-1.0, 1.0]")
        else:
            images = next(iter(dataloader))
            print(f"Images shape: {images.shape}")
            print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"Expected range: [-1.0, 1.0]")
        
        print()
        print("=" * 60)
        print("Dataset loader is working correctly!")
        print("=" * 60)
        print()
        print("You can now use this dataset for training:")
        if args.conditional:
            print(f"  python train_gan.py --mode train --data_root {args.data_root} --conditional")
        else:
            print(f"  python train_gan.py --mode train --data_root {args.data_root}")
        
    except Exception as e:
        print()
        print("=" * 60)
        print("Error occurred!")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("1. Check that your data_root path is correct")
        print("2. If using --conditional, make sure you have subdirectories:")
        print("   data_root/empty/, data_root/overlap/, data_root/clutter/")
        print("3. Make sure your images are in supported formats:")
        print("   .png, .jpg, .jpeg, .bmp, .tiff")
        print("4. Check that image files are not corrupted")