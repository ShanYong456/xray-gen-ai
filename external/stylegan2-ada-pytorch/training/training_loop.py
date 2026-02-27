# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
import random

from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from metrics import metric_main

#from training.stylegan2_adaptive_lr import StyleGAN2AdaptiveLR


#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]
    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------
def _get_opt_lr(opt):
    if opt is None or len(opt.param_groups) == 0:
        return None
    return float(opt.param_groups[0].get("lr", 0.0))

def _broadcast_lr(lr_g, lr_d, device):
    # Keep all ranks using identical LR values.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        t = torch.tensor([lr_g, lr_d], device=device, dtype=torch.float32)
        torch.distributed.broadcast(t, src=0)
        return float(t[0].item()), float(t[1].item())
    return lr_g, lr_d



def _get_main_optimizers(phases):
    g_opt = None
    d_opt = None
    for ph in phases:
        # In StyleGAN2-ADA, Gmain/Greg share the same optimizer; same for D
        if ph.name.startswith('G') and g_opt is None:
            g_opt = ph.opt
        if ph.name.startswith('D') and d_opt is None:
            d_opt = ph.opt
    return g_opt, d_opt

def _find_resume_state_path(resume_pkl):
    if resume_pkl is None:
        return None
    base = os.path.basename(resume_pkl)
    if base.startswith("network-snapshot-") and base.endswith(".pkl"):
        kimg_str = base[len("network-snapshot-"):-len(".pkl")]
        candidate = os.path.join(os.path.dirname(resume_pkl), f"training-state-{kimg_str}.pt")
        if os.path.isfile(candidate):
            return candidate
    return None

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',
    training_set_kwargs     = {},
    data_loader_kwargs      = {},
    G_kwargs                = {},
    D_kwargs                = {},
    G_opt_kwargs            = {},
    D_opt_kwargs            = {},
    augment_kwargs          = None,
    loss_kwargs             = {},
    metrics                 = [],
    random_seed             = 0,
    num_gpus                = 1,
    rank                    = 0,
    batch_size              = 4,
    batch_gpu               = 4,
    ema_kimg                = 10,
    ema_rampup              = None,
    G_reg_interval          = 4,
    D_reg_interval          = 16,
    augment_p               = 0,
    ada_target              = None,
    ada_interval            = 4,
    ada_kimg                = 500,
    total_kimg              = 25000,
    kimg_per_tick           = 4,
    image_snapshot_ticks    = 50,
    network_snapshot_ticks  = 50,
    resume_pkl              = None,
    cudnn_benchmark         = True,
    allow_tf32              = False,
    abort_fn                = None,
    progress_fn             = None,
):
    start_time = time.time()
    device = torch.device('cuda', rank)

    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)

    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    conv2d_gradfix.enabled = True
    grid_sample_gradfix.enabled = True

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(
        dataset=training_set,
        sampler=training_set_sampler,
        batch_size=batch_size//num_gpus,
        **data_loader_kwargs
    ))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    G_ema = copy.deepcopy(G).eval()

    # Resume weights from existing pickle (rank0, then DDP broadcast handles the rest).
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        img = misc.print_module_summary(G, [z, c])
        misc.print_module_summary(D, [img, c])

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device)
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    ddp_modules = dict()
    for name, module in [('G_mapping', G.mapping), ('G_synthesis', G.synthesis), ('D', D), (None, G_ema), ('augment_pipe', augment_pipe)]:
        if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
            module.requires_grad_(True)
            module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False)
            module.requires_grad_(False)
        if name is not None:
            ddp_modules[name] = module

    # Setup training phases (creates optimizers).
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, **ddp_modules, **loss_kwargs)
    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs)
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else:
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs)
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]

    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # -----------------------------
    # Progress counters (defaults)
    # -----------------------------
    cur_nimg = 0
    cur_tick = 0
    batch_idx = 0

    # -----------------------------
    # Resume FULL state (if exists)
    # -----------------------------
    resume_state_path = _find_resume_state_path(resume_pkl)
    if (resume_state_path is not None) and (rank == 0):
        print(f'Resuming full state from "{resume_state_path}"')
        st = torch.load(resume_state_path, map_location="cpu")

        cur_nimg = int(st.get("cur_nimg", cur_nimg))
        cur_tick = int(st.get("cur_tick", cur_tick))
        batch_idx = int(st.get("batch_idx", batch_idx))

        # Restore augment p
        if augment_pipe is not None and st.get("augment_p", None) is not None:
            augment_pipe.p.copy_(torch.as_tensor(st["augment_p"], device=device))

        # Restore optimizers
        g_opt, d_opt = _get_main_optimizers(phases)
        if g_opt is not None and st.get("G_opt", None) is not None:
            g_opt.load_state_dict(st["G_opt"])
            for pg in g_opt.param_groups:
                pg["lr"] = pg["lr"] * 1.0
        if d_opt is not None and st.get("D_opt", None) is not None:
            d_opt.load_state_dict(st["D_opt"])
            for pg in d_opt.param_groups:
               pg["lr"] = pg["lr"] * 0.8

        # Restore RNG
        rng = st.get("rng", {})
        if rng.get("python", None) is not None:
            random.setstate(rng["python"])
        if rng.get("numpy", None) is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("torch", None) is not None:
            torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("torch_cuda", None) is not None:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
    """
    
    # -----------------------------
    # Adaptive LR controller init
    # -----------------------------
    adaptive_lr = None
    g_opt, d_opt = _get_main_optimizers(phases)

    # IMPORTANT: pick target ranges matching StyleGAN2-ADA loss scale (based on your logs)
    adaptive_lr = StyleGAN2AdaptiveLR(
        initial_lr_g=_get_opt_lr(g_opt) or 0.0025,
        initial_lr_d=_get_opt_lr(d_opt) or 0.0025,   # will be overwritten if your D lr is tiny
        target_d_range=(0.3, 0.7),
        target_g_range=(1.2, 1.6),
        window_size=2,
        adjustment_interval=2,
        cooldown_ticks=1,
        lr_bounds_d=(0.00000001, 0.01),
        lr_bounds_g=(0.0025, 0.08),
    )

    # Sync controller’s internal lr to optimizer actual lr (esp. if you manually scaled them on resume)
    if g_opt is not None:
        adaptive_lr.lr_g = _get_opt_lr(g_opt)
    if d_opt is not None:
        adaptive_lr.lr_d = _get_opt_lr(d_opt)

    """
       

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
        save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()

    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time

    if progress_fn is not None:
        progress_fn(cur_nimg // 1000, total_kimg)

    while True:

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_c = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)

            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]

            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue

            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)

            for round_idx, (real_img, real_c, gen_z, gen_c) in enumerate(zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c)):
                sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
                gain = phase.interval
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, sync=sync, gain=gain)

            phase.module.requires_grad_(False)
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                for param in phase.module.parameters():
                    if param.grad is not None:
                        misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                phase.opt.step()

            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))

        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)

        # Save network snapshot (+ full state at SAME cadence).
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module

            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

                # Full training state
                g_opt, d_opt = _get_main_optimizers(phases)
                full_state = {
                    "cur_nimg": cur_nimg,
                    "cur_tick": cur_tick,
                    "batch_idx": batch_idx,
                    "G_opt": (g_opt.state_dict() if g_opt is not None else None),
                    "D_opt": (d_opt.state_dict() if d_opt is not None else None),
                    "augment_p": (float(augment_pipe.p.cpu()) if augment_pipe is not None else None),
                    "rng": {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    },
                }
                full_state_path = os.path.join(run_dir, f"training-state-{cur_nimg//1000:06d}.pt")
                torch.save(full_state, full_state_path)

        # Evaluate metrics.
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(
                    metric=metric,
                    G=snapshot_data['G_ema'],
                    dataset_kwargs=training_set_kwargs,
                    num_gpus=num_gpus,
                    rank=rank,
                    device=device
                )
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
        del snapshot_data

        # --- Adjust update frequency: 2G updates per 1D update ---
        # Dmain runs every 2 steps instead of every step.
        for ph in phases:
            if ph.name == 'Dmain':
                ph.interval = 2
# --------------------------------------------------------


        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)

        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        """
        # -----------------------------
        # Adaptive LR update + adjust (once per tick)
        # -----------------------------
        if adaptive_lr is not None:
            # Grab the mean losses for the tick
            d_loss = float(stats_dict["Loss/D/loss"].mean) if "Loss/D/loss" in stats_dict else None
            g_loss = float(stats_dict["Loss/G/loss"].mean) if "Loss/G/loss" in stats_dict else None

            if (d_loss is not None) and (g_loss is not None) and (g_opt is not None) and (d_opt is not None):
                adaptive_lr.update(d_loss=d_loss, g_loss=g_loss)

                # Only rank0 decides; then broadcast to other ranks to keep sync
                if rank == 0:
                    lr_g, lr_d, changed, reason = adaptive_lr.adjust_learning_rates(
                        optimizer_G=g_opt,
                        optimizer_D=d_opt,
                        verbose=True
                    )
                else:
                    lr_g = adaptive_lr.lr_g
                    lr_d = adaptive_lr.lr_d

                # Broadcast new lr to all ranks (if DDP)
                lr_g, lr_d = _broadcast_lr(lr_g, lr_d, device=device)

                # Apply broadcasted values on non-rank0 (rank0 already applied)
                if rank != 0:
                    for pg in g_opt.param_groups:
                        pg["lr"] = lr_g
                    for pg in d_opt.param_groups:
                        pg["lr"] = lr_d
                    adaptive_lr.lr_g = lr_g
                    adaptive_lr.lr_d = lr_d

                # Optional: log LR to training_stats so it appears in stats.jsonl/tensorboard
                training_stats.report0("LR/G", lr_g)
                training_stats.report0("LR/D", lr_d)

        """
        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()

        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state for next tick.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time

        if done:
            break

    if rank == 0:
        print()
        print('Exiting...')
