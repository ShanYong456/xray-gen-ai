"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

"""General-purpose training script for image-to-image translation."""

import json
import time
from pathlib import Path
import sys


from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import init_ddp, cleanup_ddp

# Add project root to PYTHONPATH
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# NEW: import FID helper
from Codes_Notebooks.Pix2Pix.fid_eval import compute_fid_for_checkpoint


def append_fid_history(checkpoints_dir, experiment_name, record):
    hist_path = Path(checkpoints_dir) / experiment_name / "fid_history.jsonl"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_path, "a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    opt = TrainOptions().parse()
    opt.device = init_ddp()

    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f"The number of training images = {dataset_size}")

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        for i, data in enumerate(dataset):
            iter_start_time = time.time()

            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, total_iters, save_result
                )

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(total_iters, losses)

            if total_iters % opt.save_latest_freq == 0:
                print(f"saving the latest model (epoch {epoch}, total_iters {total_iters})")
                save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        model.update_learning_rate()

        saved_this_epoch = False
        if epoch % opt.save_epoch_freq == 0:
            print(f"saving the model at the end of epoch {epoch}, iters {total_iters}")
            model.save_networks("latest")
            model.save_networks(epoch)
            saved_this_epoch = True

        # NEW: FID evaluation at end of epoch
        if getattr(opt, "fid_during_training", False):
            should_run_fid = (epoch % int(getattr(opt, "fid_epoch_freq", 5)) == 0)

            if should_run_fid:
                # Make sure checkpoint exists for this epoch
                if not saved_this_epoch:
                    print(f"[FID] saving checkpoint for epoch {epoch} before FID evaluation")
                    model.save_networks("latest")
                    model.save_networks(epoch)

                print(f"[FID] running FID for epoch {epoch} ...")
                try:
                    fid_result = compute_fid_for_checkpoint(
                        args=opt,
                        epoch=epoch,
                        phase=getattr(opt, "fid_phase", "val"),
                        max_images=int(getattr(opt, "fid_max_images", 200)),
                        keep_images=bool(getattr(opt, "fid_keep_images", False)),
                        work_dir=getattr(opt, "fid_work_dir", "fid_eval_runs"),
                        debug_every=int(getattr(opt, "fid_debug_every", 10)),
                    )

                    append_fid_history(opt.checkpoints_dir, opt.name, fid_result)

                    fid_value = fid_result.get("fid", None)
                    if fid_value is not None:
                        print(f"[FID] epoch {epoch}: {fid_value:.6f}")
                    else:
                        print(f"[FID] epoch {epoch}: unavailable")

                except Exception as e:
                    print(f"[FID] epoch {epoch} failed: {e}")

        print(
            f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} "
            f"\t Time Taken: {time.time() - epoch_start_time:.0f} sec"
        )

    cleanup_ddp()

"""

Train

python external/pix2pix/train.py \
  --dataroot datasets/non-contraband_grid9 \
  --name non-contraband_pix2pix \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 3 --output_nc 3 \
  --netG unet_256 \
  --preprocess none --load_size 512 --crop_size 512 \
  --batch_size 1 --no_flip \
  --lambda_L1 10 \
  --n_epochs 600 --n_epochs_decay 200
  

  Train more

python external/pix2pix/train.py \
  --dataroot datasets/non-contraband_grid9 \
  --name non-contraband_pix2pix \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 3 --output_nc 3 \
  --netG unet_256 \
  --preprocess none --load_size 512 --crop_size 512 \
  --batch_size 1 --no_flip \
  --lambda_L1 10 \
  --continue_train --epoch latest \
  --n_epochs 600 --n_epochs_decay 200

 Train V2:
python external/pix2pix/train.py \
  --dataroot datasets/non_contraband_V1 \
  --name non_contraband_pix2pix_V1 \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --preprocess none --load_size 512 --crop_size 512 \
  --netG unet_256 \
  --netD n_layers --n_layers_D 4 \
  --input_nc 3 --output_nc 3 \
  --gan_mode lsgan \
  --lambda_L1 80 \
  --norm instance \
  --lr 0.0002 \
  --n_epochs 20 --n_epochs_decay 50 \
  --continue_train --epoch latest \
  --batch_size 1 \
  --no_flip

  python external/pix2pix/train.py \
  --dataroot datasets/contraband_metal_VV1\
  --name contraband_metal_pix2pix_VV2 \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --preprocess resize_and_crop --load_size 1024 --crop_size 1024 \
  --netG unet_256 \
  --netD n_layers --n_layers_D 4 \
  --input_nc 3 --output_nc 3 \
  --gan_mode lsgan \
  --lambda_L1 60 \
  --norm instance \
  --lr 0.0002 \
  --n_epochs 150 --n_epochs_decay 150 \
  --batch_size 1 \
  --no_flip


  V3:
    python external/pix2pix/train.py \
  --dataroot datasets/non_contraband_V1 \
  --name non_contraband_pix2pix_phys_fixDull_v1 \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 6 --output_nc 3 \
  --netG unet_256 --norm instance \
  --gan_mode lsgan \
  --lambda_L1 10 \
  --use_delta_comp --use_masked_l1 --lambda_bg 3.0 \
  --use_masked_gan --gan_bg_keep 0.35 \
  --gan_mask_dilate --gan_dilate_px 7 \
  --use_fm --lambda_fm 10 \
  --delta_scale 0.7 --od_gain 4.0 \
  --use_gamma --gamma 2.2 \
  --compose_eps 1e-6 \
  --empty_dir data/interim/GAN/Empty \
  --preprocess none --load_size 1024 --crop_size 1024 \
  --lr_G 0.0004  --lr_D 0.000008\
  --n_epochs 150 --n_epochs_decay 150 \
  --continue_train --epoch latest \
  --batch_size 1 --no_flip



 python external/pix2pix/train.py \
  --dataroot datasets/contraband_metal_VV1 \
  --name contraband_metal_pix2pix_phys_fixDull_v3-1\
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 6 --output_nc 3 \
  --netG unet_256 --norm instance \
  --gan_mode lsgan \
  --lambda_L1 10 \
  --use_delta_comp --use_masked_l1 --lambda_bg 3.0 \
  --use_masked_gan --gan_bg_keep 0.35 \
  --gan_mask_dilate --gan_dilate_px 7 \
  --use_fm --lambda_fm 10 \
  --delta_scale 0.7 --od_gain 4.0 \
  --use_gamma --gamma 2.2 \
  --compose_eps 1e-6 \
  --empty_dir data/interim/GAN/Empty \
  --preprocess none --load_size 1024 --crop_size 1024 \
  --lr_G 0.0002  --lr_D 0.00004\
  --n_epochs 150 --n_epochs_decay 150 \
  --continue_train --epoch latest \
  --batch_size 1 --no_flip



  python external/pix2pix/train.py \
  --dataroot datasets/Shampoo \
  --name Shampoo_pix2pix_phys_fixDull_v1\
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 6 --output_nc 3 \
  --netG unet_256 --norm instance \
  --gan_mode lsgan \
  --lambda_L1 10 \
  --use_delta_comp --use_masked_l1 --lambda_bg 3.0 \
  --use_masked_gan --gan_bg_keep 0.35 \
  --gan_mask_dilate --gan_dilate_px 7 \
  --use_fm --lambda_fm 10 \
  --delta_scale 0.7 --od_gain 4.0 \
  --use_gamma --gamma 2.2 \
  --compose_eps 1e-6 \
  --empty_dir data/interim/GAN/Empty \
  --preprocess none --load_size 1024 --crop_size 1024 \
  --lr_G 0.0002  --lr_D 0.00002\
  --n_epochs 150 --n_epochs_decay 150 \
  --batch_size 1 --no_flip

   python external/pix2pix/train.py \
  --dataroot datasets/Shampoo \
  --name Shampoo_pix2pix_phys_v1 \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 6 --output_nc 3 \
  --netG unet_256 --norm instance \
  --gan_mode lsgan \
  --lambda_L1 60 \
  --use_delta_comp \
  --use_masked_l1 --lambda_bg 1.5 \
  --empty_dir data/interim/GAN/Empty \
  --preprocess none --load_size 1024 --crop_size 1024 \
  --n_epochs 150 --n_epochs_decay 150 \
  --batch_size 1 --no_flip --no_dropout

  python external/pix2pix/train.py \
  --dataroot datasets/Shampoo \
  --name Shampoo_pix2pix_V1 \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 6 --output_nc 3 \
  --netG unet_256 --norm instance \
  --gan_mode lsgan \
  --lambda_L1 60 --use_masked_l1 --lambda_bg 1.5 \
  --use_delta_comp --delta_scale 1.0 --compose_eps 1e-6 \
  --empty_dir data/interim/GAN/Empty \
  --preprocess none --load_size 1024 --crop_size 1024 \
  --n_epochs 150 --n_epochs_decay 150 \
  --lr_G 0.0002 --lr_D 0.0002\
  --continue_train --epoch latest \
  --batch_size 1 --no_flip --no_dropout

  python external/pix2pix/train.py \
  --dataroot datasets/Shampoo_Blade \
  --name Shampoo_Blade_pix2pix_V1 \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 6 --output_nc 3 \
  --netG unet_256 --norm instance \
  --gan_mode lsgan \
  --lambda_L1 60 --use_masked_l1 --lambda_bg 1.5 \
  --use_delta_comp --delta_scale 1.0 --compose_eps 1e-6 \
  --empty_dir data/interim/GAN/Empty \
  --preprocess none --load_size 1024 --crop_size 1024 \
  --n_epochs 17 --n_epochs_decay 150 \
  --lr_G 0.0002 --lr_D 0.0002\
  --continue_train --epoch latest \
  --batch_size 1 --no_flip --no_dropout

   python external/pix2pix/train.py \
  --dataroot datasets/Shampoo_Blade \
  --name Shampoo_Blade_pix2pix_V2 \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 6 --output_nc 3 \
  --netG unet_256 --norm instance \
  --gan_mode lsgan \
  --lambda_L1 80 --use_masked_l1 --lambda_bg 0.1 \
  --use_delta_comp --compose_eps 1e-6 \
  --delta_positive --delta_scale 0.3 --delta_max 4.0 \
  --od_gamma 0.7 --mask_nc 3 --lambda_delta_bg 5.0 \
  --empty_dir data/interim/GAN/Empty \
  --preprocess none --load_size 1024 --crop_size 1024 \
  --n_epochs 17 --n_epochs_decay 150 \
  --lr_G 0.0002 --lr_D 0.0001 \
  --batch_size 1 --no_flip --no_dropout
  

  python external/pix2pix/train.py \
  --dataroot datasets/Shampoo_Blade \
  --name Shampoo_Blade_pix2pix_V2_detailV1 \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 6 --output_nc 3 \
  --netG unet_256 --norm instance \
  --gan_mode lsgan \
  --lambda_L1 10 --use_masked_l1 --lambda_bg 1.5 \
  --use_delta_comp --delta_positive --delta_max 10 --delta_scale 0.7 \
  --use_delta_supervision --lambda_delta 150 \
  --lambda_delta_bg 2 \
  --use_grad_loss --lambda_grad 15 \
  --use_delta_grad_loss --lambda_delta_grad 15 \
  --use_lap_loss --lambda_lap 15 \
  --use_ssim_loss --lambda_ssim 1 \
  --use_region_stats --lambda_stats 1 \
  --lr 0.00015 --beta1 0.5 \
  --preprocess none --load_size 1024 --crop_size 1024 \
  --batch_size 1 --no_flip --no_dropout \
  --empty_dir data/interim/GAN/Empty


  python external/pix2pix/train.py \
  --dataroot datasets/Shampoo_Blade \
  --name Shampoo_Blade_pix2pix_V2_detailV2 \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 6 --output_nc 3 \
  --netG unet_256 --norm instance \
  --gan_mode lsgan \
  --lambda_L1 8 \
  --use_masked_l1 --lambda_bg 1.5 \
  --use_delta_comp --delta_positive --delta_max 10 --delta_scale 0.7 \
  --use_delta_supervision --lambda_delta 120 \
  --lambda_delta_bg 2 \
  --use_grad_loss --lambda_grad 15 --use_tray_mask\
  --use_delta_grad_loss --lambda_delta_grad 15 --tray_shift_max_px 400 \
  --use_lap_loss --lambda_lap 15 --tray_shift_iters 1\
  --use_ssim_loss --lambda_ssim 5 --empty_dir data/interim/GAN/Empty \
  --use_region_stats --lambda_stats 5 --batch_size 1 --no_flip --no_dropout\
  --use_soft_mask --preprocess none --load_size 1024 --crop_size 1024  \
  --mask_blur_ksize 5 --lr 0.00015 --beta1 0.5\
  --mask_noise_std 0.02 --tray_mask_autoshift\
  --tray_mask_path data/interim/GAN/Empty/Mask/2026-01-21_10-36-28-447_traymask.png

  
  python external/pix2pix/train.py \
  --dataroot datasets/Shampoo_Blade \
  --name Shampoo_Blade_pix2pix_V2_detailV2 \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 6 --output_nc 3 \
  --netG unet_256 --norm instance \
  --gan_mode lsgan \
  --lambda_L1 8 \
  --use_masked_l1 --lambda_bg 1.5 \
  --use_delta_comp --delta_positive --delta_max 10 --delta_scale 0.7 \
  --use_delta_supervision --lambda_delta 120 \
  --lambda_delta_bg 2 \
  --use_tray_mask \
  --use_grad_loss --lambda_grad 15 \
  --use_delta_grad_loss --lambda_delta_grad 15 --tray_bbox_margin 2 --tray_mask_erode_px 0 \
  --use_lap_loss --lambda_lap 15 --tray_shift_max_px 400 \
  --use_ssim_loss --lambda_ssim 5 --lr 0.00015 --beta1 0.5 --tray_mask_autoshift\
  --use_region_stats --lambda_stats 5 --batch_size 1 --no_flip --no_dropout\
  --use_soft_mask --preprocess none --load_size 1024 --crop_size 1024\
  --mask_blur_ksize 5 --mask_noise_std 0.02 --empty_dir data/interim/GAN/Empty\
  --tray_mask_path data/interim/GAN/Empty/Mask/2026-01-21_10-36-28-447_traymask.png

  python external/pix2pix/train.py \
  --dataroot datasets/Shampoo_Blade \
  --name Shampoo_Blade_pix2pix_V2_detailV3 \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 6 --output_nc 3 \
  --netG unet_256 --norm instance \
  --gan_mode lsgan \
  --lambda_L1 8 \
  --use_masked_l1 --lambda_bg 1.5 \
  --use_delta_comp --delta_positive --delta_max 10 --delta_scale 0.7 \
  --use_delta_supervision --lambda_delta 120 --lr 0.00015 --beta1 0.5 \
  --lambda_delta_bg 2 --mask_blur_ksize 5 --mask_noise_std 0.02 \
  --use_tray_mask --preprocess none --load_size 1024 --crop_size 1024 \
  --use_grad_loss --lambda_grad 15 \
  --use_delta_grad_loss --lambda_delta_grad 15 \
  --use_lap_loss --lambda_lap 15 --tray_mask_autoshift --tray_bbox_margin 5 \
  --use_ssim_loss --lambda_ssim 5 --tray_mask_dilate_px 0 --tray_obj_dilate_px 2 \
  --use_region_stats --lambda_stats 5 --tray_nudge_iters 8 --tray_nudge_max_step 20 \
  --batch_size 1 --no_flip --no_dropout --tray_shift_max_px 400 \
  --use_soft_mask --empty_dir data/interim/GAN/Empty \
  --tray_mask_path data/interim/GAN/Empty/Mask/2026-01-21_10-36-28-447_traymask.png

python external/pix2pix/train.py \
--dataroot datasets/Shampoo_Blade --name Shampoo_Blade_pix2pix_SyntecticMaskV3 \
--model pix2pix --dataset_mode aligned --direction AtoB --input_nc 6 --output_nc 3 \
--netG unet_256 --norm instance --gan_mode lsgan \
--lambda_L1 8 --use_masked_l1 --lambda_bg 1.5 \
--use_delta_comp --delta_positive --delta_max 10 --delta_scale 0.7 \
--use_delta_supervision --lambda_delta 120 --lambda_delta_bg 2 \
--use_grad_loss --lambda_grad 15 --use_delta_grad_loss --lambda_delta_grad 15 \
--use_lap_loss --lambda_lap 8 --use_ssim_loss --lambda_ssim 2 --use_region_stats --lambda_stats 2 \
--use_tray_mask --tray_mask_autoshift --tray_obj_dilate_px 5 --tray_bbox_margin 2 --tray_mask_dilate_px 3 \
--tray_nudge_iters 8 --tray_nudge_max_step 20 --tray_shift_max_px 400 \
--synthetic_prob 0.15 --synthetic_no_overlap --synthetic_min_items 1 --synthetic_max_items 2 \
--synthetic_scale_min 0.85 --synthetic_scale_max 1.15 --synthetic_rot_max 25 \
--lambda_syn_tv 0.2 --lambda_syn_mag 0.05 \
--empty_dir data/interim/GAN/Empty \
--tray_mask_path data/interim/GAN/Empty/Mask/2026-01-21_10-36-28-447_traymask.png \
--cutout_dir data/raw/Shampoo_Blade/Cropped \
--preprocess none --load_size 1024 --crop_size 1024 \
--batch_size 1 --lr 0.00015 --beta1 0.5 \
--n_epochs 150 --n_epochs_decay 150 --num_threads 0
  
python external/pix2pix/train.py \
  --dataroot datasets/Shampoo_Blade \
  --name Shampoo_Blade_pix2pix_AppearanceV2 \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 6 --output_nc 3 --class_nc 1 --appearance_nc 1 --thickness_nc 1 \
  --use_thickness_channel --use_appearance_channel --return_instance_masks \
  --netG unet_256 --norm instance --gan_mode lsgan \
  --lambda_L1 6 --use_masked_l1 --lambda_bg 1.5 \
  --use_delta_comp --delta_positive --delta_max 3 --delta_scale 0.35 \
  --use_delta_supervision --lambda_delta 70 --lambda_instance_delta 4 --lambda_delta_bg 2 \
  --use_grad_loss --lambda_grad 8 --use_delta_grad_loss --lambda_delta_grad 6 \
  --use_lap_loss --lambda_lap 4 --use_ssim_loss --lambda_ssim 1.5 --use_region_stats --lambda_stats 1 \
  --use_tray_mask --tray_mask_autoshift --tray_obj_dilate_px 5 --tray_bbox_margin 2 --tray_mask_dilate_px 3 \
  --tray_nudge_iters 8 --tray_nudge_max_step 20 --tray_shift_max_px 400 \
  --synthetic_prob 0.10 --synthetic_same_class_prob 0.6 --synthetic_no_overlap \
  --synthetic_min_items 1 --synthetic_max_items 2 \
  --synthetic_scale_min 0.9 --synthetic_scale_max 1.1 --synthetic_rot_max 20 \
  --lambda_syn_tv 0.15 --lambda_syn_mag 0.04 \
  --empty_dir data/interim/GAN/Empty \
  --tray_mask_path data/interim/GAN/Empty/Mask/2026-01-21_10-36-28-447_traymask.png \
  --cutout_dir data/raw/Shampoo_Blade/Cropped \
  --preprocess none --load_size 1024 --crop_size 1024 \
  --batch_size 1 --lr 0.0001 --beta1 0.5 \
  --n_epochs 150 --n_epochs_decay 150 --num_threads 0


  python external/pix2pix/train.py \
  --dataroot datasets/Shampoo \
  --name Shampoo_pix2pix_StructCond_V1 \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 8 --output_nc 3 --class_nc 1 --thickness_nc 1 \
  --use_thickness_channel --use_edge_channel --use_coord_channels --return_instance_masks \
  --netG unet_256 --norm instance --gan_mode lsgan \
  --lambda_L1 7 --use_masked_l1 --lambda_bg 1.5 \
  --use_delta_comp --delta_positive --delta_max 3 --delta_scale 0.8 \
  --use_delta_supervision --lambda_delta 120 --lambda_instance_delta 8 --lambda_delta_bg 2 \
  --use_grad_loss --lambda_grad 12 --use_delta_grad_loss --lambda_delta_grad 14 \
  --use_lap_loss --lambda_lap 8 --use_ssim_loss --lambda_ssim 1.5 --use_region_stats --lambda_stats 2 \
  --use_tray_mask --tray_mask_autoshift --tray_obj_dilate_px 5 --tray_bbox_margin 2 --tray_mask_dilate_px 3 \
  --tray_nudge_iters 8 --tray_nudge_max_step 20 --tray_shift_max_px 400 \
  --synthetic_prob 0.35 --synthetic_same_class_prob 0.6 --synthetic_no_overlap \
  --synthetic_min_items 1 --synthetic_max_items 2 \
  --synthetic_scale_min 0.75 --synthetic_scale_max 1.25 --synthetic_rot_max 45 \
  --lambda_syn_tv 0.15 --lambda_syn_mag 0.04 \
  --empty_dir data/interim/GAN/Empty \
  --tray_mask_path data/interim/GAN/Empty/Mask/2026-01-21_10-36-28-447_traymask.png \
  --cutout_dir data/raw/Shampoo/Cropped \
  --preprocess none --load_size 1024 --crop_size 1024 \
  --batch_size 1 --lr 0.0001 --beta1 0.5 \
  --n_epochs 150 --n_epochs_decay 150 --num_threads 0

 python external/pix2pix/train.py \
  --dataroot datasets/Shampoo \
  --name Shampoo_pix2pix_StructCond_V2 \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 9 --output_nc 3 --class_nc 1 --thickness_nc 1 --appearance_nc 1 \
  --use_thickness_channel --use_edge_channel --use_coord_channels --use_appearance_channel --return_instance_masks \
  --netG unet_256 --norm instance --gan_mode lsgan \
  --lambda_L1 7 --use_masked_l1 --lambda_bg 1.5 \
  --use_delta_comp --delta_positive --delta_max 3 --delta_scale 0.8 \
  --use_delta_supervision --lambda_delta 120 --lambda_instance_delta 8 --lambda_delta_bg 2 \
  --use_grad_loss --lambda_grad 12 --use_delta_grad_loss --lambda_delta_grad 14 \
  --use_lap_loss --lambda_lap 8 --use_ssim_loss --lambda_ssim 1.5 --use_region_stats --lambda_stats 2 \
  --use_tray_mask --tray_mask_autoshift --tray_obj_dilate_px 5 --tray_bbox_margin 2 --tray_mask_dilate_px 3 \
  --tray_nudge_iters 8 --tray_nudge_max_step 20 --tray_shift_max_px 400 \
  --synthetic_prob 0.10 --synthetic_same_class_prob 0.6 --synthetic_no_overlap \
  --synthetic_min_items 1 --synthetic_max_items 2 \
  --synthetic_scale_min 0.75 --synthetic_scale_max 1.25 --synthetic_rot_max 45 \
  --appearance_dropout 0.6 \
  --lambda_syn_tv 0.15 --lambda_syn_mag 0.04 \
  --empty_dir data/interim/GAN/Empty \
  --tray_mask_path data/interim/GAN/Empty/Mask/2026-01-21_10-36-28-447_traymask.png \
  --cutout_dir data/raw/Shampoo/Cropped \
  --pretrained_netG checkpoints/Shampoo_pix2pix_StructCond_V1/latest_net_G.pth \
  --preprocess none --load_size 1024 --crop_size 1024 \
  --batch_size 1 --lr 0.00005 --beta1 0.5 \
  --n_epochs 80 --n_epochs_decay 80 --num_threads 0

  
python external/pix2pix/train.py \
  --dataroot datasets/Shampoo_nobackground \
  --name Shampoo_NOBGR_pix2pix_StructCond_V1_Stage2 \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 6 --output_nc 3 \
  --netG unet_256 --netD n_layers --n_layers_D 4 --norm instance \
  --preprocess none --load_size 0 --crop_size 0 --no_flip \
  --batch_size 1 --pool_size 0 --gan_mode lsgan \
  --lr 1e-5 --beta1 0.5 --n_epochs 80 --n_epochs_decay 80 \
  --class_nc 1 --thickness_nc 1 --appearance_nc 1 \
  --use_thickness_channel --use_edge_channel --use_coord_channels --use_appearance_channel \
  --return_instance_masks --mask_thr 0.05 \
  --use_masked_l1 --lambda_L1 30 --lambda_bg 0 \
  --use_grad_loss --lambda_grad 10 \
  --use_lap_loss --lambda_lap 6 \
  --use_ssim_loss --lambda_ssim 3 \
  --use_region_stats --lambda_stats 3 \
  --d_label_smooth 0.1 \
  --pad_to_canvas --canvas_w 1024 --canvas_h 1536 --canvas_fill 0 \
  --pretrained_netG checkpoints/Shampoo_NOBGR_pix2pix_StructCond_V1_Stage2/latest_net_G.pth \
  --synthetic_prob 0.0 \
  --appearance_zero_prob 0.45 \
  --appearance_weak_prob 0.35 \
  --appearance_proto_prob 0.15

python external/pix2pix/train.py \
  --dataroot datasets/Shampoo_nobackground \
  --name Shampoo_NOBGR_pix2pix_StructCond_V1_Stage5_Syn \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 6 --output_nc 3 \
  --netG unet_256 --netD n_layers --n_layers_D 4 --norm instance \
  --preprocess none --load_size 0 --crop_size 0 --no_flip \
  --batch_size 1 --pool_size 0 --gan_mode lsgan \
  --lr 1e-5 --beta1 0.5 --n_epochs 80 --n_epochs_decay 80 \
  --class_nc 1 --thickness_nc 1 --appearance_nc 1 \
  --use_thickness_channel --use_edge_channel --use_coord_channels --use_appearance_channel \
  --return_instance_masks --mask_thr 0.05 \
  --use_masked_l1 --lambda_L1 30 --lambda_bg 0 \
  --use_grad_loss --lambda_grad 10 \
  --use_lap_loss --lambda_lap 6 \
  --use_ssim_loss --lambda_ssim 3 \
  --use_region_stats --lambda_stats 3 \
  --d_label_smooth 0.1 --synthetic_min_items 1 --synthetic_max_items 3\
  --pad_to_canvas --canvas_w 1024 --canvas_h 1536 --canvas_fill 0 \
  --synthetic_prob 0.7 \
  --synthetic_no_overlap \
  --appearance_zero_prob 0.85 \
  --appearance_weak_prob 0.125 \
  --appearance_proto_prob 0.025 \
  --tray_mask_dir data/interim/Empty/masks_viz \
  --cutout_dir data/raw/Shampoo_nobackground/Cropped \
  --continue_train --epoch latest



STAGE A SHAMPOO

python external/pix2pix/train.py \
  --dataroot datasets/Shampoo_nobackground \
  --name Shampoo_NOBGR_pix2pix_StructCond_V1_Stage8_Syn \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 5 --output_nc 3 \
  --netG unet_256 --netD n_layers --n_layers_D 4 --norm instance \
  --preprocess none --load_size 0 --crop_size 0 --no_flip \
  --batch_size 1 --pool_size 0 --gan_mode lsgan \
  --lr 1e-5 --beta1 0.5 --n_epochs 100 --n_epochs_decay 100 \
  --class_nc 1 --thickness_nc 1 \
  --use_thickness_channel --use_edge_channel --use_coord_channels \
  --return_instance_masks --mask_thr 0.05 \
  --use_masked_l1 --lambda_L1 30 --lambda_bg 0 \
  --use_grad_loss --lambda_grad 10 \
  --use_lap_loss --lambda_lap 6 \
  --use_ssim_loss --lambda_ssim 3 \
  --use_region_stats --lambda_stats 3 \
  --d_label_smooth 0.1 \
  --appearance_zero_prob 0.95 \
  --appearance_weak_prob 0.025 \
  --appearance_proto_prob 0.025 \
  --synthetic_min_items 1 --synthetic_max_items 5 \
  --pad_to_canvas --canvas_w 1024 --canvas_h 1536 --canvas_fill 0 \
  --synthetic_prob 0.95 --synthetic_scale_min 1.0 --synthetic_scale_max 1.0 \
  --synthetic_no_overlap --synthetic_rot_min 0 --synthetic_rot_max 0 \
  --cutout_dir data/raw/Shampoo_nobackground/Cropped_Library \
   --continue_train --epoch latest




  
python external/pix2pix/train.py \
  --dataroot datasets/Shampoo_EmptyTray_Iso \
  --name Shampoo_NOBGR_pix2pix_StructCond_V1_Stage9_SynTray \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 5 --output_nc 3 \
  --netG unet_256 --netD n_layers --n_layers_D 4 --norm instance \
  --preprocess none --load_size 0 --crop_size 0 --no_flip \
  --batch_size 1 --pool_size 0 --gan_mode lsgan \
  --lr 1e-5 --beta1 0.5 --n_epochs 100 --n_epochs_decay 100 \
  --class_nc 1 --thickness_nc 1 \
  --use_thickness_channel --use_edge_channel --use_coord_channels \
  --return_instance_masks --mask_thr 0.05 \
  --use_masked_l1 --lambda_L1 30 --lambda_bg 0 \
  --use_grad_loss --lambda_grad 10 \
  --use_lap_loss --lambda_lap 6 \
  --use_ssim_loss --lambda_ssim 3 \
  --use_region_stats --lambda_stats 3 \
  --d_label_smooth 0.1 \
  --appearance_zero_prob 0.95 \
  --appearance_weak_prob 0.025 \
  --appearance_proto_prob 0.025 \
  --use_tray_mask \
  --tray_mask_dir data/interim/Empty/masks_viz \
  --synthetic_min_items 1 --synthetic_max_items 5 \
  --pad_to_canvas --canvas_w 1584 --canvas_h 1152 --canvas_fill 0 \
  --synthetic_prob 0.95 --synthetic_scale_min 1.0 --synthetic_scale_max 1.0 \
  --synthetic_no_overlap --synthetic_rot_min 0 --synthetic_rot_max 0 \
  --cutout_dir data/raw/Shampoo_nobackground/Cropped_Library \
  --pretrained_netG checkpoints/Shampoo_NOBGR_pix2pix_StructCond_V1_Stage8_Syn/latest_net_G.pth

STAGE 1 
python external/pix2pix/train.py \
  --dataroot datasets/Shampoo_EmptyTray_Iso \
  --name Shampoo_NOBGR_pix2pix_StructCond_V1_Stage10_SynTray_1024 \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 6 --output_nc 3 \
  --netG unet_256 --netD n_layers --n_layers_D 4 --norm instance \
  --preprocess none --load_size 0 --crop_size 0 --no_flip \
  --batch_size 1 --pool_size 0 --gan_mode lsgan \
  --lr 5e-6 --beta1 0.5 --n_epochs 50 --n_epochs_decay 50 \
  --class_nc 2 --thickness_nc 1 \
  --use_thickness_channel --use_edge_channel --use_coord_channels \
  --return_instance_masks --mask_thr 0.05 \
  --use_masked_l1 --lambda_L1 30 --lambda_bg 0 \
  --use_grad_loss --lambda_grad 10 \
  --use_lap_loss --lambda_lap 6 \
  --use_ssim_loss --lambda_ssim 3 \
  --use_region_stats --lambda_stats 3 \
  --d_label_smooth 0.1 \
  --use_tray_mask \
  --tray_mask_dir datasets/Shampoo_EmptyTray_Iso/tray_masks/train \
  --tray_mask_thr 0 \
  --tray_cc_close_px 2 \
  --tray_mask_dilate_px 0 \
  --shampoo_horizontal_shift_only \
  --shampoo_max_horizontal_shift 150 \
  --shampoo_max_vertical_shift 0 \
  --synthetic_min_items 1 --synthetic_max_items 2 \
  --synthetic_place_tries 30 \
  --synthetic_item_retries 4 \
  --synthetic_erode_px 2 \
  --synthetic_fallback_shrink 0.85 \
  --pad_to_canvas --canvas_w 1024 --canvas_h 1024 --canvas_fill 0 \
  --synthetic_prob 0.30 --synthetic_scale_min 0.85 --synthetic_scale_max 0.85 --tray_scale 1.0 \
  --synthetic_no_overlap --synthetic_rot_min 0 --synthetic_rot_max 40 \
  --cutout_dir data/raw/Shampoo_nobackground/Cropped_Library \
  --continue_train --epoch latest


STAGE2
 python external/pix2pix/train.py \
  --dataroot datasets/Shampoo_EmptyTray_Iso \
  --name Shampoo_NOBGR_pix2pix_StructCond_V1_Stage10_SynTray_1024 \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 6 --output_nc 3 \
  --netG unet_256 --netD n_layers --n_layers_D 4 --norm instance \
  --preprocess none --load_size 0 --crop_size 0 --no_flip \
  --batch_size 1 --pool_size 0 --gan_mode lsgan \
  --lr 5e-6 --beta1 0.5 --n_epochs 50 --n_epochs_decay 50 \
  --class_nc 2 --thickness_nc 1 \
  --use_thickness_channel --use_edge_channel --use_coord_channels \
  --return_instance_masks --mask_thr 0.05 \
  --use_masked_l1 --lambda_L1 30 --lambda_bg 0 \
  --use_grad_loss --lambda_grad 10 \
  --use_lap_loss --lambda_lap 6 \
  --use_ssim_loss --lambda_ssim 3 \
  --use_region_stats --lambda_stats 3 \
  --d_label_smooth 0.1 \
  --use_tray_mask \
  --tray_mask_dir datasets/Shampoo_EmptyTray_Iso/tray_masks/train \
  --tray_mask_thr 0 \
  --tray_cc_close_px 2 \
  --tray_mask_dilate_px 0 \
  --shampoo_horizontal_shift_only \
  --shampoo_max_horizontal_shift 150 \
  --shampoo_max_vertical_shift 0 \
  --synthetic_min_items 1 --synthetic_max_items 2 \
  --synthetic_place_tries 30 \
  --synthetic_item_retries 4 \
  --synthetic_erode_px 2 \
  --synthetic_fallback_shrink 0.85 \
  --pad_to_canvas --canvas_w 1024 --canvas_h 1024 --canvas_fill 0 \
  --synthetic_prob 0.85 --synthetic_scale_min 0.85 --synthetic_scale_max 0.85 --tray_scale 1.0 \
  --synthetic_no_overlap --synthetic_rot_min 0 --synthetic_rot_max 40 \
  --cutout_dir data/raw/Shampoo_nobackground/Cropped_Library \
  --continue_train --epoch latest
  
  --synthetic_prob
  --pretrained_netG checkpoints/Shampoo_NOBGR_pix2pix_StructCond_V1_Stage9_Syn/latest_net_G.pth
  
   --continue_train --epoch latest

STAGE 3

python external/pix2pix/train.py \
  --dataroot datasets/SHAMPOOWITHTRAY \
  --name Shampoo_NOBGR_pix2pix_StructCond_V1_Stage11_SynTray_1024 \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 6 --output_nc 3 \
  --netG unet_256 --netD n_layers --n_layers_D 4 --norm instance \
  --preprocess none --load_size 0 --crop_size 0 --no_flip \
  --batch_size 1 --pool_size 0 --gan_mode lsgan \
  --lr 5e-6 --beta1 0.5 --n_epochs 100 --n_epochs_decay 100 \
  --class_nc 2 --thickness_nc 1 \
  --use_thickness_channel --use_edge_channel --use_coord_channels \
  --return_instance_masks --mask_thr 0.05 \
  --use_masked_l1 --lambda_L1 30 --lambda_bg 0 \
  --use_grad_loss --lambda_grad 10 \
  --use_lap_loss --lambda_lap 6 \
  --use_ssim_loss --lambda_ssim 3 \
  --use_region_stats --lambda_stats 3 \
  --d_label_smooth 0.1 \
  --use_tray_mask \
  --tray_mask_dir datasets/SHAMPOOWITHTRAY/matched_masks/train/tray \
  --tray_mask_thr 0 \
  --tray_cc_close_px 2 \
  --tray_mask_dilate_px 0 \
  --shampoo_horizontal_shift_only \
  --shampoo_max_horizontal_shift 150 \
  --shampoo_max_vertical_shift 0 \
  --synthetic_min_items 1 --synthetic_max_items 2 \
  --synthetic_place_tries 30 \
  --synthetic_item_retries 4 \
  --synthetic_erode_px 2 \
  --synthetic_fallback_shrink 0.85 \
  --pad_to_canvas --canvas_w 1024 --canvas_h 1024 --canvas_fill 0 \
  --synthetic_prob 0 --synthetic_scale_min 0.85 --synthetic_scale_max 0.85 --tray_scale 1.0 \
  --synthetic_no_overlap --synthetic_rot_min 0 --synthetic_rot_max 40 \
  --cutout_dir data/raw/Shampoo_nobackground/Cropped_Library \
  --continue_train --epoch latest



  Stage 4

python external/pix2pix/train.py \
--dataroot datasets/SHAMPOOWITHTRAY \
--name Shampoo_NOBGR_pix2pix_StructCond_V1_Stage12_SynTray_1024 \
--model pix2pix --dataset_mode aligned --direction AtoB \
--input_nc 6 --output_nc 3 --class_nc 2 --thickness_nc 1 \
--netG unet_256 --netD n_layers --n_layers_D 4 --norm instance \
--preprocess none --load_size 0 --crop_size 0 --no_flip \
--batch_size 1 --pool_size 0 --gan_mode lsgan \
--lr 5e-6 --beta1 0.5 --n_epochs 100 --n_epochs_decay 100 \
--use_thickness_channel --use_edge_channel --use_coord_channels --return_instance_masks \
--use_masked_l1 --lambda_L1 30 --lambda_bg 0 \
--use_grad_loss --lambda_grad 10 --use_lap_loss --lambda_lap 6 \
--use_ssim_loss --lambda_ssim 3 --use_region_stats --lambda_stats 3 \
--d_label_smooth 0.1 --d_update_ratio 2 \
--use_tray_mask --tray_mask_dir datasets/SHAMPOOWITHTRAY/matched_masks/train/tray \
--synthetic_min_items 1 --synthetic_max_items 1 \
--synthetic_place_tries 10 --synthetic_item_retries 4 --synthetic_erode_px 2 \
--synthetic_prob 0.35 --synthetic_no_overlap \
--pad_to_canvas --canvas_w 1792 --canvas_h 1280 \
--cutout_dir data/raw/Shampoo_nobackground/Cropped_Library \
--continue_train --epoch latest --synthetic_scale_min 2.0 --synthetic_scale_max 2.0


python external/pix2pix/train.py \
--dataroot datasets/SHAMPOOWITHTRAY \
--name Shampoo_NOBGR_pix2pix_StructCond_V1_Stage14TEST \
--model pix2pix --dataset_mode aligned --direction AtoB \
--input_nc 6 --output_nc 3 --class_nc 2 --thickness_nc 1 \
--netG unet_256 --netD n_layers --n_layers_D 4 --norm instance \
--preprocess none --load_size 0 --crop_size 0 --no_flip \
--batch_size 1 --pool_size 0 --gan_mode lsgan \
--lr 5e-6 --beta1 0.5 --n_epochs 100 --n_epochs_decay 100 \
--use_thickness_channel --use_edge_channel --use_coord_channels --return_instance_masks \
--use_masked_l1 --lambda_L1 45 --lambda_bg 0 \
--use_grad_loss --lambda_grad 10 --use_lap_loss --lambda_lap 6 \
--use_ssim_loss --lambda_ssim 3 --use_region_stats --lambda_stats 3 \
--d_label_smooth 0.1 --d_update_ratio 2 \
--use_tray_mask --tray_mask_dir datasets/SHAMPOOWITHTRAY/matched_masks/train/tray \
--synthetic_min_items 1 --synthetic_max_items 1 \
--synthetic_place_tries 30 --synthetic_item_retries 6 --synthetic_erode_px 2 \
--synthetic_prob 0.5 --synthetic_no_overlap \
--pad_to_canvas --canvas_w 1024 --canvas_h 1024 \
--cutout_dir data/raw/Shampoo_nobackground/Cropped_Library \
--continue_train --epoch latest --synthetic_scale_min 0.65 --synthetic_scale_max 0.65



python external/pix2pix/train.py \
--dataroot datasets/SHAMPOOWITHTRAY \
--name Shampoo_NOBGR_pix2pix_StructCond_V1_Stage15TEST \
--model pix2pix --dataset_mode aligned --direction AtoB \
--input_nc 6 --output_nc 3 --class_nc 2 --thickness_nc 1 \
--netG unet_256 --netD n_layers --n_layers_D 4 --norm instance \
--preprocess none --load_size 0 --crop_size 0 --no_flip \
--batch_size 1 --pool_size 0 --gan_mode lsgan \
--lr 5e-6 --beta1 0.5 --n_epochs 100 --n_epochs_decay 100 \
--use_thickness_channel --use_edge_channel --use_coord_channels --return_instance_masks \
--use_masked_l1 --lambda_L1 45 --lambda_bg 0 \
--use_grad_loss --lambda_grad 10 --use_lap_loss --lambda_lap 6 \
--use_ssim_loss --lambda_ssim 3 --use_region_stats --lambda_stats 3 \
--d_label_smooth 0.1 --d_update_ratio 2 \
--use_tray_mask --tray_mask_dir datasets/SHAMPOOWITHTRAY/matched_masks/train/tray \
--synthetic_min_items 1 --synthetic_max_items 1 \
--synthetic_place_tries 30 --synthetic_item_retries 6 --synthetic_erode_px 2 \
--synthetic_prob 0.65 --synthetic_no_overlap \
--pad_to_canvas --canvas_w 1024 --canvas_h 1024 \
--cutout_dir data/raw/Shampoo_nobackground/Cropped_Library \
--continue_train --epoch latest \
--synthetic_scale_min 0.75 --synthetic_scale_max 0.65 \
--fid_during_training \
--fid_epoch_freq 40 \
--fid_phase train \
--fid_max_images 358 \
--fid_work_dir fid_eval_runs


BLADE AND SHAMPOO ISO:

python external/pix2pix/train.py \
  --dataroot datasets/SHAMPOOBLADEWITHTRAY \
  --name Shampoo_NOBGR_pix2pix_StructCond_V1_Stage17_BladeMaskSyn \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 7 --output_nc 3 --class_nc 3 --thickness_nc 1 \
  --netG unet_256 --netD n_layers --n_layers_D 4 --norm instance \
  --preprocess none --load_size 0 --crop_size 0 --no_flip \
  --batch_size 1 --pool_size 0 --gan_mode lsgan \
  --lr 3e-6 --beta1 0.5 --n_epochs 150 --n_epochs_decay 150 \
  --use_thickness_channel --use_edge_channel --use_coord_channels --return_instance_masks \
  --use_masked_l1 --lambda_L1 45 --lambda_bg 0 \
  --use_grad_loss --lambda_grad 10 --use_lap_loss --lambda_lap 6 \
  --use_ssim_loss --lambda_ssim 3 --use_region_stats --lambda_stats 3 \
  --d_label_smooth 0.1 --d_update_ratio 2 \
  --use_tray_mask --tray_mask_dir datasets/SHAMPOOBLADEWITHTRAY/matched_masks/train/tray \
  --synthetic_min_items 1 --synthetic_max_items 1 \
  --synthetic_place_tries 30 --synthetic_item_retries 6 --synthetic_erode_px 2 \
  --synthetic_prob 0 --synthetic_no_overlap \
  --pad_to_canvas --canvas_w 1024 --canvas_h 1024 \
  --cutout_dir data/raw/Shampoo_nobackground/Cropped_Library \
  --synthetic_blade_mask_dir datasets/SHAMPOOBLADEWITHTRAY/matched_masks/train/blade \
  --synthetic_scale_min 0.65 --synthetic_scale_max 0.75 \
  --continue_train --epoch latest \
  --fid_during_training \
  --fid_epoch_freq 40 \
  --fid_phase train \
  --fid_max_images 300 \
  --fid_work_dir fid_eval_runs




  BLADE AND SHAMPOO TGT:

python external/pix2pix/train.py \
  --dataroot datasets/SHAMPOOBLADEWITHTRAY_TGT \
  --name Shampoo_NOBGR_pix2pix_StructCond_V1_Stage18_BladeMaskSyn \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 7 --output_nc 3 --class_nc 3 --thickness_nc 1 \
  --netG unet_256 --netD n_layers --n_layers_D 4 --norm instance \
  --preprocess none --load_size 0 --crop_size 0 --no_flip \
  --batch_size 1 --pool_size 0 --gan_mode lsgan \
  --lr 3e-6 --beta1 0.5 --n_epochs 150 --n_epochs_decay 150 \
  --use_thickness_channel --use_edge_channel --use_coord_channels --return_instance_masks \
  --use_masked_l1 --lambda_L1 45 --lambda_bg 0 \
  --use_grad_loss --lambda_grad 10 --use_lap_loss --lambda_lap 6 \
  --use_ssim_loss --lambda_ssim 3 --use_region_stats --lambda_stats 3 \
  --d_label_smooth 0.1 --d_update_ratio 2 \
  --use_tray_mask --tray_mask_dir datasets/SHAMPOOBLADEWITHTRAY_TGT/matched_masks/train/tray \
  --synthetic_min_items 1 --synthetic_max_items 1 \
  --synthetic_place_tries 30 --synthetic_item_retries 6 --synthetic_erode_px 2 \
  --synthetic_prob 0 --synthetic_no_overlap \
  --pad_to_canvas --canvas_w 1024 --canvas_h 1024 \
  --cutout_dir data/raw/Shampoo_nobackground/Cropped_Library \
  --synthetic_blade_mask_dir datasets/SHAMPOOBLADEWITHTRAY_TGT/matched_masks/train/blade \
  --synthetic_scale_min 0.65 --synthetic_scale_max 0.75 \
  --continue_train --epoch latest \
  --fid_during_training \
  --fid_epoch_freq 40 \
  --fid_phase train \
  --fid_max_images 500 \
  --fid_work_dir fid_eval_runs




    BLADE AND SHAMPOO COMPLETE:

python external/pix2pix/train.py \
  --dataroot datasets/SHAMPOOBLADEWITHTRAY_COMPLETE \
  --name Shampoo_NOBGR_pix2pix_StructCond_V1_Stage20_COMPLETESyn \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 7 --output_nc 3 --class_nc 3 --thickness_nc 1 \
  --netG unet_256 --netD n_layers --n_layers_D 4 --norm instance \
  --preprocess none --load_size 0 --crop_size 0 --no_flip \
  --batch_size 1 --pool_size 0 --gan_mode lsgan \
  --lr 3e-6 --beta1 0.5 --n_epochs 150 --n_epochs_decay 150 \
  --use_thickness_channel --use_edge_channel --use_coord_channels --return_instance_masks \
  --use_masked_l1 --lambda_L1 45 --lambda_bg 0 \
  --use_grad_loss --lambda_grad 10 --use_lap_loss --lambda_lap 6 \
  --use_ssim_loss --lambda_ssim 3 --use_region_stats --lambda_stats 3 \
  --d_label_smooth 0.1 --d_update_ratio 2 \
  --use_tray_mask --tray_mask_dir datasets/SHAMPOOBLADEWITHTRAY_COMPLETE/matched_masks/train/tray \
  --synthetic_min_items 1 --synthetic_max_items 1 \
  --synthetic_place_tries 30 --synthetic_item_retries 6 --synthetic_erode_px 2 \
  --synthetic_prob 0.35 --synthetic_no_overlap \
  --pad_to_canvas --canvas_w 1024 --canvas_h 1024 \
  --cutout_dir data/raw/Shampoo_nobackground/Cropped_Library \
  --synthetic_blade_mask_dir datasets/SHAMPOOBLADEWITHTRAY_COMPLETE/matched_masks/train/blade \
  --synthetic_scale_min 0.65 --synthetic_scale_max 0.75 \
  --continue_train --epoch latest \
  --fid_during_training \
  --fid_epoch_freq 40 \
  --fid_phase train \
  --fid_max_images 500 \
  --fid_work_dir fid_eval_runs


  python external/pix2pix/train.py \
  --dataroot datasets/SHAMPOOBLADEWITHTRAY_COMPLETE \
  --name Shampoo_NOBGR_pix2pix_StructCond_V1_Stage23_COMPLETESyn \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 7 --output_nc 3 --class_nc 3 --thickness_nc 1 \
  --netG unet_256 --netD n_layers --n_layers_D 4 --norm instance \
  --preprocess none --load_size 0 --crop_size 0 --no_flip \
  --batch_size 1 --pool_size 0 --gan_mode lsgan \
  --lr 3e-6 --beta1 0.5 --n_epochs 150 --n_epochs_decay 150 \
  --use_thickness_channel --use_edge_channel --use_coord_channels --return_instance_masks \
  --use_masked_l1 --lambda_L1 25 --lambda_bg 0 \
  --use_grad_loss --lambda_grad 5 \
  --use_lap_loss --lambda_lap 2 \
  --use_ssim_loss --lambda_ssim 2 \
  --d_label_smooth 0.1 \
  --syn_gan_weight 1.0 \
  --d_update_ratio 1 \
  --use_tray_mask --tray_mask_dir datasets/SHAMPOOBLADEWITHTRAY_COMPLETE/matched_masks/train/tray \
  --synthetic_prob 0.5 \
  --synthetic_combo_mode random \
  --synthetic_prob_shampoo_only 0.25 \
  --synthetic_prob_blade_only 0.25 \
  --synthetic_prob_pair_no_overlap 0.25 \
  --synthetic_prob_pair_overlap 0.25 \
  --synthetic_place_tries 60 --synthetic_item_retries 10 --synthetic_erode_px 2 \
  --pad_to_canvas --canvas_w 1024 --canvas_h 1024 \
  --cutout_dir data/raw/Shampoo_nobackground/Cropped_Library \
  --synthetic_blade_mask_dir datasets/SHAMPOOBLADEWITHTRAY_COMPLETE/matched_masks/train/blade \
  --synthetic_scale_min 0.65 --synthetic_scale_max 0.65 \
  --continue_train --epoch latest


  
  python external/pix2pix/train.py \
  --dataroot datasets/SHAMPOOBLADEWITHTRAY_COMPLETE \
  --name Shampoo_NOBGR_pix2pix_StructCond_V1_Stage24_COMPLETESyn \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 7 --output_nc 3 --class_nc 3 --thickness_nc 1 \
  --netG unet_256 --netD n_layers --n_layers_D 4 --norm instance \
  --preprocess none --load_size 0 --crop_size 0 --no_flip \
  --batch_size 1 --pool_size 0 --gan_mode lsgan \
  --lr 3e-6 --beta1 0.5 --n_epochs 150 --n_epochs_decay 150 \
  --use_thickness_channel --use_edge_channel --use_coord_channels --return_instance_masks \
  --use_masked_l1 --lambda_L1 25 --lambda_bg 0 \
  --use_grad_loss --lambda_grad 5 \
  --use_lap_loss --lambda_lap 2 \
  --use_ssim_loss --lambda_ssim 2 \
  --d_label_smooth 0.1 \
  --syn_gan_weight 0.5 \
  --d_update_ratio 1 \
  --use_tray_mask --tray_mask_dir datasets/SHAMPOOBLADEWITHTRAY_COMPLETE/matched_masks/train/tray \
  --synthetic_prob 0.5 \
  --synthetic_combo_mode random \
  --synthetic_prob_shampoo_only 0.25 \
  --synthetic_prob_blade_only 0.25 \
  --synthetic_prob_pair_no_overlap 0.25 \
  --synthetic_prob_pair_overlap 0.25 \
  --synthetic_place_tries 100 --synthetic_item_retries 16 --synthetic_erode_px 2 \
  --synthetic_sort_large_first \
  --pad_to_canvas --canvas_w 1024 --canvas_h 1024 \
  --cutout_dir data/raw/Shampoo_nobackground/Cropped_Library \
  --synthetic_blade_mask_dir datasets/SHAMPOOBLADEWITHTRAY_COMPLETE/matched_masks/train/blade \
  --synthetic_scale_min 0.60 --synthetic_scale_max 0.72 \
  --synthetic_rot_min -8 --synthetic_rot_max 8 \
  --continue_train --epoch latest



  # WITHOUT EMPTY TRAY
   python external/pix2pix/train.py \
  --dataroot datasets/Shampoo \
  --name Shampoo_pix2pix_plain_v1 \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 3 --output_nc 3 \
  --netG unet_256 --norm instance \
  --gan_mode lsgan \
  --lambda_L1 10 \
  --use_fm --lambda_fm 10 \
  --preprocess none --load_size 1024 --crop_size 1024 \
  --lr_G 0.0002 --lr_D 0.0002 \
  --n_epochs 150 --n_epochs_decay 150 \
  --batch_size 1 --no_flip

 

Test

python external/pix2pix/test.py \
  --dataroot datasets/non-contraband_grid9 \
  --name non-contraband_pix2pix \
  --model pix2pix \
  --dataset_mode aligned \
  --direction AtoB \
  --input_nc 3 \
  --output_nc 3 \
  --preprocess none --load_size 512 --crop_size 512 \
  --netG unet_256 \


  
"""