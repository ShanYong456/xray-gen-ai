"""
Post-process pix2pix output to remove white halos around objects.

The white halo appears at object boundaries. This script:
1. Detects the object boundary from the input mask
2. Extends the mask slightly inward 
3. Blends the output using the extended mask to remove edge artifacts
"""

import cv2
import numpy as np
from pathlib import Path


def remove_halo(fake_B_path: str, mask_A_path: str, out_path: str, 
                erode_px: int = 3, blend_width: int = 5):
    """
    Remove halo from generated image.
    
    Args:
        fake_B_path: Path to generated fake_B image
        mask_A_path: Path to input mask (A) 
        out_path: Output path for halo-removed image
        erode_px: Pixels to erode object mask inward (larger = more halo removed)
        blend_width: Smooth transition width
    """
    # Read images
    fake_B = cv2.imread(str(fake_B_path))
    if fake_B is None:
        raise FileNotFoundError(f"Could not read: {fake_B_path}")
    
    # Read the corresponding input mask
    mask_A = cv2.imread(str(mask_A_path))
    if mask_A is None:
        raise FileNotFoundError(f"Could not read: {mask_A_path}")
    
    # Extract object mask from A (any non-black pixels)
    # Shampoo_Blade uses blue=2 or green=1, so check all channels
    obj_mask = np.any(mask_A > 10, axis=2).astype(np.uint8) * 255
    
    # Erode the mask inward to create a "safe zone"
    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px*2+1, erode_px*2+1))
        safe_zone = cv2.erode(obj_mask, kernel, iterations=1)
    else:
        safe_zone = obj_mask.copy()
    
    # Create smooth transition mask
    if blend_width > 0:
        # Dilate back slightly to create gradient
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blend_width*2+1, blend_width*2+1))
        blend_zone = cv2.dilate(safe_zone, kernel, iterations=1)
        
        # Create smooth gradient (0 outside, 1 inside)
        transition = blend_zone.astype(np.float32) / 255.0
        transition = cv2.GaussianBlur(transition, (blend_width*2+1, blend_width*2+1), 0)
    else:
        transition = safe_zone.astype(np.float32) / 255.0
    
    # Find empty region (bg in test)
    # For inference with empty tray, read the right side of AB pair
    fake_B_float = fake_B.astype(np.float32) / 255.0
    
    # Blend: keep fake_B inside safe zone, fade outside
    for c in range(3):
        fake_B[:,:,c] = (transition[:,:,np.newaxis] * fake_B[:,:,c].astype(np.float32) + 
                         (1 - transition[:,:,np.newaxis]) * 128).astype(np.uint8)
    
    # Save result
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), fake_B)
    print(f"Halo-removed image saved to: {out_path}")


def process_results(
    model_name: str,
    epoch: str = "latest",
    results_base: Path = Path("results"),
    datasets_base: Path = Path("datasets"),
    erode_px: int = 5,
    blend_width: int = 8,
):
    """
    Process all generated images from a pix2pix result directory.
    
    Args:
        model_name: Name of the pix2pix model
        epoch: Epoch to process (default "latest")
        results_base: Base results directory
        datasets_base: Base datasets directory
        erode_px: How much to erode (larger = more halo removal)
        blend_width: Blend width for smooth transition
    """
    results_dir = results_base / model_name / f"test_{epoch}"
    images_dir = results_dir / "images"
    
    if not images_dir.exists():
        print(f"Results directory not found: {images_dir}")
        return
    
    # Find all fake_B images
    fake_B_files = sorted(images_dir.glob("*_fake_B.png"))
    if not fake_B_files:
        print(f"No fake_B images found in: {images_dir}")
        return
    
    print(f"Found {len(fake_B_files)} generated images")
    
    # For each fake_B, find corresponding mask A
    for fake_B_path in fake_B_files:
        # Try to find mask from the test dataset
        # Pattern: fake_B might be from AB pair like "gen_real_scene_count_Shampoo_seed128_real_A.png"
        # or just "000001_fake_B.png"
        
        # Look for AB file in the same directory
        ab_file = images_dir / fake_B_path.name.replace("_fake_B", "")
        if ab_file.exists():
            # Extract A from AB pair
            AB = cv2.imread(str(ab_file))
            if AB.shape[1] == 2048:  # Concatenated AB
                mask_A = AB[:, :1024, :]
            else:
                continue
        else:
            # Try to find in dataset directories
            # This is a fallback
            continue
        
        # Remove halo
        out_path = images_dir / fake_B_path.name.replace("_fake_B", "_fake_B_dehalo")
        remove_halo_with_mask(fake_B_path, mask=mask_A, out_path=out_path, 
                             erode_px=erode_px, blend_width=blend_width)


def remove_halo_with_mask(
    fake_B_path: str,
    mask: np.ndarray,
    out_path: str,
    erode_px: int = 5,
    blend_width: int = 8,
):
    """Remove halo using a pre-loaded mask array."""
    fake_B = cv2.imread(str(fake_B_path))
    if fake_B is None:
        raise FileNotFoundError(f"Could not read: {fake_B_path}")
    
    # Extract object mask from provided mask
    obj_mask = np.any(mask > 10, axis=2).astype(np.uint8) * 255
    
    # Erode the mask inward
    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px*2+1, erode_px*2+1))
        safe_zone = cv2.erode(obj_mask, kernel, iterations=1)
    else:
        safe_zone = obj_mask.copy()
    
    # Create smooth transition
    if blend_width > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blend_width*2+1, blend_width*2+1))
        blend_zone = cv2.dilate(safe_zone, kernel, iterations=1)
        transition = blend_zone.astype(np.float32) / 255.0
        transition = cv2.GaussianBlur(transition, (blend_width*2+1, blend_width*2+1), 0)
    else:
        transition = safe_zone.astype(np.float32) / 255.0
    
    # Blend result
    result = fake_B.copy().astype(np.float32)
    for c in range(3):
        result[:,:,c] = transition * result[:,:,c] + (1 - transition) * 200
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), result)
    print(f"Halo-removed image saved to: {out_path}")


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake_B", type=str, help="Path to fake_B image from pix2pix")
    ap.add_argument("--mask_A", type=str, help="Path to input mask A")
    ap.add_argument("--out", type=str, default="output_dehalo.png", help="Output path")
    ap.add_argument("--erode_px", type=int, default=5, help="Erosion pixels (larger = more halo removal)")
    ap.add_argument("--blend_width", type=int, default=8, help="Blend width for smooth transition")
    args = ap.parse_args()
    
    if args.fake_B and args.mask_A:
        remove_halo(args.fake_B, args.mask_A, args.out, 
                   erode_px=args.erode_px, blend_width=args.blend_width)
    else:
        print("Usage: python halo_remover.py --fake_B <path> --mask_A <path> --out <path>")
