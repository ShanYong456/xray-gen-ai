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

import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import init_ddp, cleanup_ddp


if __name__ == "__main__":
    opt = TrainOptions().parse()  # get training options
    opt.device = init_ddp()
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print(f"The number of training images = {dataset_size}")

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()
        # Set epoch for DistributedSampler
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, total_iters, save_result)

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(total_iters, losses)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print(f"saving the latest model (epoch {epoch}, total_iters {total_iters})")
                save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        model.update_learning_rate()  # update learning rates at the end of every epoch

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print(f"saving the model at the end of epoch {epoch}, iters {total_iters}")
            model.save_networks("latest")
            model.save_networks(epoch)

        print(f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time:.0f} sec")

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
  --name non_contraband_pix2pix_physV3 \
  --model pix2pix --dataset_mode aligned --direction AtoB \
  --input_nc 6 --output_nc 3 \
  --netG unet_256 --norm instance \
  --gan_mode lsgan --lambda_L1 60 \
  --use_delta_comp --use_masked_l1 --lambda_bg 3.0 \
  --empty_dir data/interim/GAN/Empty \
  --preprocess none --load_size 1024 --crop_size 1024 \
  --n_epochs 150 --n_epochs_decay 150 \
  --batch_size 1 --no_flip --no_dropout



 python external/pix2pix/train.py \
  --dataroot datasets/contraband_metal_VV1 \
  --name contraband_metal_pix2pix_physV4_1024_od4_gamma \
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
  --lr_G 0.0002 --lr_D 0.0001 \
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