import os
import sys
import argparse
from pathlib import Path
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from notebooks.CDGan.model import (
    Generator, Discriminator,
    ConditionalGenerator, ConditionalDiscriminator
)
from notebooks.CDGan.dataset import Get_dataloader


# ============================================================
# AdaptiveHyperparameterTuner (kept as-is from your version)
# ============================================================
class AdaptiveHyperparameterTuner:
    """
    Improved adaptive hyperparameter tuner with stability controls
    """
    def __init__(self, initial_lr_g, initial_lr_d, initial_n_critic, window_size=10):
        self.lr_g = initial_lr_g
        self.lr_d = initial_lr_d
        self.n_critic = initial_n_critic

        self.window_size = window_size
        self.g_loss_history = deque(maxlen=window_size)
        self.d_loss_history = deque(maxlen=window_size)

        self.lr_adjust_factor = 0.85
        self.min_lr = 1e-6
        self.max_lr = 1e-3
        self.max_n_critic = 5
        self.min_n_critic = 1

        self.target_d_g_ratio = 1.0
        self.adjustment_threshold = 0.5

        self.cooldown_epochs = 3
        self.epochs_since_adjustment = 99
        self.adjustment_count = 0
        self.max_adjustments_per_window = 3

        self.prev_adjustment_direction = None
        self.oscillation_count = 0

    def update(self, g_loss, d_loss):
        self.g_loss_history.append(g_loss)
        self.d_loss_history.append(d_loss)
        self.epochs_since_adjustment += 1

    def should_adjust(self):
        has_history = len(self.g_loss_history) >= self.window_size
        past_cooldown = self.epochs_since_adjustment >= self.cooldown_epochs

        if self.epochs_since_adjustment >= self.window_size:
            self.adjustment_count = 0

        under_limit = self.adjustment_count < self.max_adjustments_per_window
        return has_history and past_cooldown and under_limit

    def calculate_metrics(self):
        if len(self.g_loss_history) < self.window_size:
            return None

        avg_g_loss = np.mean(self.g_loss_history)
        avg_d_loss = np.mean(self.d_loss_history)

        g_trend = np.polyfit(range(self.window_size), list(self.g_loss_history), 1)[0]
        d_trend = np.polyfit(range(self.window_size), list(self.d_loss_history), 1)[0]

        loss_ratio = avg_d_loss / (avg_g_loss + 1e-8)

        g_variance = np.var(self.g_loss_history)
        d_variance = np.var(self.d_loss_history)

        return {
            'avg_g_loss': avg_g_loss,
            'avg_d_loss': avg_d_loss,
            'g_trend': g_trend,
            'd_trend': d_trend,
            'loss_ratio': loss_ratio,
            'g_variance': g_variance,
            'd_variance': d_variance
        }

    def detect_oscillation(self, current_direction):
        if self.prev_adjustment_direction is not None:
            if self.prev_adjustment_direction != current_direction:
                self.oscillation_count += 1
            else:
                self.oscillation_count = max(0, self.oscillation_count - 1)

        self.prev_adjustment_direction = current_direction
        return self.oscillation_count >= 2

    def adjust_hyperparameters(self):
        metrics = self.calculate_metrics()
        if metrics is None:
            return self.lr_g, self.lr_d, self.n_critic, False

        adjustment_made = False
        old_values = (self.lr_g, self.lr_d, self.n_critic)

        avg_g = metrics['avg_g_loss']
        avg_d = metrics['avg_d_loss']
        g_trend = metrics['g_trend']
        d_trend = metrics['d_trend']
        loss_ratio = metrics['loss_ratio']
        g_var = metrics['g_variance']
        d_var = metrics['d_variance']

        adjustment_direction = None

        if avg_d < 0.2 and avg_g > 2.2:
            print(f"\nEMERGENCY: D dominating (D={avg_d:.3f}, G={avg_g:.3f})")
            adjustment_direction = 'emergency_weaken_d'
            self.lr_d = max(self.lr_d * 0.5, self.min_lr)
            self.lr_g = min(self.lr_g * 1.5, self.max_lr)
            self.n_critic = self.min_n_critic
            adjustment_made = True

        elif avg_d < 0.3 and avg_g > 1.65:
            print(f"\nD too strong (D={avg_d:.3f}, G={avg_g:.3f})")
            adjustment_direction = 'weaken_d'
            self.lr_d = max(self.lr_d * 0.7, self.min_lr)
            self.lr_g = min(self.lr_g * 1.3, self.max_lr)
            self.n_critic = max(self.n_critic - 1, self.min_n_critic)
            adjustment_made = True

        elif avg_g < 0.7 and avg_d > 0.9:
            print(f"\nG too strong (D={avg_d:.3f}, G={avg_g:.3f})")
            adjustment_direction = 'weaken_g'
            self.lr_d = min(self.lr_d * 1.3, self.max_lr)
            self.lr_g = max(self.lr_g * 0.7, self.min_lr)
            self.n_critic = min(self.n_critic + 1, self.max_n_critic)
            adjustment_made = True

        elif g_trend > 0.25 and d_trend > 0.25 and g_var < 2.0 and d_var < 2.0:
            print(f"\nBoth losses rising (instability)")
            adjustment_direction = 'stabilize'
            self.lr_g = max(self.lr_g * 0.9, self.min_lr)
            self.lr_d = max(self.lr_d * 0.9, self.min_lr)
            adjustment_made = True

        elif g_trend < -0.1 and d_trend < -0.1 and avg_g < 0.3 and avg_d < 0.3:
            print(f"\nMode collapse risk (both low and dropping)")
            adjustment_direction = 'prevent_collapse'
            self.lr_g = min(self.lr_g * 1.05, self.max_lr)
            self.lr_d = min(self.lr_d * 1.05, self.max_lr)
            adjustment_made = True

        elif abs(loss_ratio - self.target_d_g_ratio) > self.adjustment_threshold:
            if g_var > 2.0 or d_var > 2.0:
                print(f"\n⏸️  Skipping: high variance (G_var={g_var:.3f}, D_var={d_var:.3f})")
                return self.lr_g, self.lr_d, self.n_critic, False

            if loss_ratio < self.target_d_g_ratio - self.adjustment_threshold:
                print(f"\nD weaker (ratio={loss_ratio:.3f})")
                adjustment_direction = 'strengthen_d'
                self.lr_d = min(self.lr_d * 1.15, self.max_lr)
                self.n_critic = min(self.n_critic + 1, self.max_n_critic)
                adjustment_made = True
            elif loss_ratio > self.target_d_g_ratio + self.adjustment_threshold:
                print(f"\n D stronger (ratio={loss_ratio:.3f})")
                adjustment_direction = 'weaken_d'
                self.lr_d = max(self.lr_d * 0.85, self.min_lr)
                self.n_critic = max(self.n_critic - 1, self.min_n_critic)
                adjustment_made = True

        if adjustment_made and adjustment_direction != 'emergency_weaken_d':
            if self.detect_oscillation(adjustment_direction):
                print(f"Oscillation detected → more conservative")
                self.lr_adjust_factor = 0.95
                self.cooldown_epochs = 5
                self.adjustment_threshold = 0.7

        if adjustment_made:
            self.epochs_since_adjustment = 0
            self.adjustment_count += 1

            print(f"   Adjustments:")
            print(f"   lr_g: {old_values[0]:.6f} → {self.lr_g:.6f}")
            print(f"   lr_d: {old_values[1]:.6f} → {self.lr_d:.6f}")
            print(f"   n_critic: {old_values[2]} → {self.n_critic}")

        return self.lr_g, self.lr_d, self.n_critic, adjustment_made


# ============================================================
# Helpers: live plot + auto LR controller
# ============================================================
def force_optimizer_lr(optimizer, new_lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = float(new_lr)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def init_live_plot():
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("GAN Loss (epoch avg)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    (line_d,) = ax.plot([], [], label="D loss")
    (line_g,) = ax.plot([], [], label="G loss")
    ax.legend()
    fig.tight_layout()
    return fig, ax, line_d, line_g

def update_live_plot(fig, ax, line_d, line_g, xs, d_hist, g_hist):
    # ---- Safety: keep lengths identical ----
    n = min(len(xs), len(d_hist), len(g_hist))
    if n == 0:
        return
    xs_ = xs[:n]
    d_  = d_hist[:n]
    g_  = g_hist[:n]

    line_d.set_data(xs_, d_)
    line_g.set_data(xs_, g_)

    ax.relim()
    ax.autoscale_view()

    fig.canvas.draw()
    fig.canvas.flush_events()


def auto_lr_controller(
    d_loss_avg, g_loss_avg,
    optimizer_G, optimizer_D,
    target_d=(0.40, 0.60),
    target_g=(1.30, 1.70),

    # bounds
    lr_g_bounds=(8e-5, 5e-3),
    lr_d_bounds=(2e-6, 2e-4),

    # scheduling
    cooldown_epochs=1,
    epoch_1based=1,
    last_adjust_epoch=0,
    interval_epochs=5,

    # stability - IMPROVED DEFAULTS
    gain=0.35,                  # Further reduced (was 0.45)
    max_mult_per_update=1.03,   # Reduced to 3% per update (was 4%)
    min_mult_per_update=0.97,   # Raised to 3% down (was 4%)
    ema_beta=0.85,              # Increased smoothing
    dominance_deadband=0.12,    # Increased deadband
    dominance_cap=0.25,         # Further reduced cap (was 0.30)

    # emergency (hard guardrails) - made more conservative
    d_too_low=0.15,             # D dominating - very low threshold
    g_too_high=2.2,             # Raised threshold significantly
    d_too_high=1.2,             # D too weak / broken
    g_too_low=0.9,             # Lowered
):
    """
    FIXED LR controller that properly handles D domination.
    
    Key insight: When D is dominating (low D loss, high G loss), we should:
    1. NEVER increase D's learning rate
    2. DECREASE D's learning rate OR keep it stable
    3. INCREASE G's learning rate
    
    The previous version had a critical flaw: it would still increase D's LR
    when D was already strong, making the problem worse.
    """

    # ---- interval gate ----
    if interval_epochs is not None and interval_epochs > 1:
        if (epoch_1based % interval_epochs) != 0:
            lr_g = optimizer_G.param_groups[0]["lr"]
            lr_d = optimizer_D.param_groups[0]["lr"]
            return lr_g, lr_d, False, last_adjust_epoch, f"interval({interval_epochs})"

    # ---- cooldown gate ----
    if (epoch_1based - last_adjust_epoch) < cooldown_epochs:
        lr_g = optimizer_G.param_groups[0]["lr"]
        lr_d = optimizer_D.param_groups[0]["lr"]
        return lr_g, lr_d, False, last_adjust_epoch, "cooldown"

    lr_g0 = float(optimizer_G.param_groups[0]["lr"])
    lr_d0 = float(optimizer_D.param_groups[0]["lr"])

    d_lo, d_hi = target_d
    g_lo, g_hi = target_g

    # -------------------------
    # 1) Emergency guardrails
    # -------------------------
    if (d_loss_avg < d_too_low and g_loss_avg > g_too_high):
        lr_d = clamp(lr_d0 * 0.92, lr_d_bounds[0], lr_d_bounds[1])
        lr_g = clamp(lr_g0 * 1.06, lr_g_bounds[0], lr_g_bounds[1])
        force_optimizer_lr(optimizer_D, lr_d)
        force_optimizer_lr(optimizer_G, lr_g)
        why = f"EMERGENCY D-dominating (D<{d_too_low} & G>{g_too_high})"
        return lr_g, lr_d, True, epoch_1based, why

    if (d_loss_avg > d_too_high and g_loss_avg < g_too_low):
        lr_d = clamp(lr_d0 * 1.15, lr_d_bounds[0], lr_d_bounds[1])
        lr_g = clamp(lr_g0 * 0.88, lr_g_bounds[0], lr_g_bounds[1])
        force_optimizer_lr(optimizer_D, lr_d)
        force_optimizer_lr(optimizer_G, lr_g)
        why = f"EMERGENCY D-weak (D>{d_too_high} & G<{g_too_low})"
        return lr_g, lr_d, True, epoch_1based, why

    # -------------------------
    # 2) CRITICAL FIX: Direct assessment of who's winning
    # -------------------------
    # Instead of complex dominance calculation, directly check:
    # - Is D too strong? (loss < 0.25)
    # - Is G too weak? (loss > 2.0)
    
    d_too_strong = d_loss_avg < d_lo
    g_too_strong = g_loss_avg < g_lo
    d_needs_help = d_loss_avg > d_hi
    g_needs_help = g_loss_avg > g_hi
    
    # Initialize multipliers
    mult_g = 1.0
    mult_d = 1.0
    action = ""
    
    # -------------------------
    # 3) MAIN CONTROL LOGIC
    # -------------------------
    
    # CASE 1: D is dominating (D too strong AND G too weak)
    if d_too_strong and g_needs_help:
        # CRITICAL: When D is dominating, we MUST weaken D and strengthen G
        # Calculate gap size to determine adjustment magnitude
        d_gap = (d_lo - d_loss_avg) / d_lo  # How far below target (0 to 1)
        g_gap = (g_loss_avg - g_hi) / g_hi  # How far above target (0 to inf)
        
        # More aggressive the bigger the gap
        strength = min(d_gap + g_gap, 0.30)  # Cap at 30%
        
        mult_d = 1.0 - strength  # Reduce D by up to 30%
        mult_g = 1.0 + (strength * 2.0)            # Increase G by up to 60%
        action = f"D-dominating [dgap={d_gap:.2f} ggap={g_gap:.2f}]"
    
    # CASE 2: D is weak AND G is strong
    elif d_needs_help and g_too_strong:
        # G is dominating, strengthen D and weaken G
        d_gap = (d_loss_avg - d_hi) / d_hi
        g_gap = (g_lo - g_loss_avg) / g_lo
        
        strength = min(d_gap + g_gap, 0.12)
        
        mult_d = 1.0 + strength
        mult_g = 1.0 - strength
        action = f"G-dominating [dgap={d_gap:.2f} ggap={g_gap:.2f}]"
    
    # CASE 3: D is too strong (but G not critically weak yet)
    elif d_too_strong:
        # Gently weaken D, gently strengthen G
        strength = min((d_lo - d_loss_avg) / d_lo, 0.05)
        mult_d = 1.0 - strength
        mult_g = 1.0 + (strength * 0.5)
        action = f"D-too-strong [str={strength:.3f}]"
    
    # CASE 4: D needs help
    elif d_needs_help:
        strength = min((d_loss_avg - d_hi) / d_hi, 0.08)
        mult_d = 1.0 + strength
        mult_g = 1.0 - (strength * 0.7)
        action = f"D-needs-help [str={strength:.3f}]"
    
    # CASE 5: G needs help
    elif g_needs_help:
        strength = min((g_loss_avg - g_hi) / g_hi, 0.08)  # positive when G too high
        strength = max(strength, 0.0)
        mult_g = 1.0 + strength
        mult_d = 1.0 - (strength * 0.7)
        action = f"G-needs-help [str={strength:.3f}]"

    
    # CASE 6: Both in acceptable range
    else:
        # Fine-tuning based on target bands
        def outside_band_error(x, lo, hi):
            w = (hi - lo) + 1e-8
            if x < lo:
                return (x - lo) / w
            if x > hi:
                return (x - hi) / w
            return 0.0
        
        d_err = outside_band_error(d_loss_avg, d_lo, d_hi)
        g_err = outside_band_error(g_loss_avg, g_lo, g_hi)
        
        if abs(d_err) < 0.1 and abs(g_err) < 0.1:
            return lr_g0, lr_d0, False, last_adjust_epoch, "in-band-stable"
        
        # Gentle adjustments
        mult_g = 1.0 + (g_err * -0.02)  # If G high, reduce lr
        mult_d = 1.0 + (d_err * -0.02)  # If D high, reduce lr
        action = f"fine-tune [d_err={d_err:+.2f} g_err={g_err:+.2f}]"
    
    # -------------------------
    # 4) Apply safety caps
    # -------------------------
    mult_g = clamp(mult_g, min_mult_per_update, max_mult_per_update)
    mult_d = clamp(mult_d, min_mult_per_update, max_mult_per_update)
    
    # CRITICAL SAFETY: Never increase D's LR when it's already strong
    if d_loss_avg < 0.20 and mult_d > 1.0:
        mult_d = 1.0
        action += " [D-strong-block-increase]"
    
    # CRITICAL SAFETY: Never decrease D's LR when it's already weak
    if d_loss_avg > 0.80 and mult_d < 1.0:
        mult_d = 1.0
        action += " [D-weak-block-decrease]"

    # -------------------------
    # 5) Apply with bounds
    # -------------------------
    lr_g = clamp(lr_g0 * mult_g, lr_g_bounds[0], lr_g_bounds[1])
    lr_d = clamp(lr_d0 * mult_d, lr_d_bounds[0], lr_d_bounds[1])

    force_optimizer_lr(optimizer_G, lr_g)
    force_optimizer_lr(optimizer_D, lr_d)

    why = (
        f"{action} | d={d_loss_avg:.4f} g={g_loss_avg:.4f} "
        f"| mult_g={mult_g:.4f} mult_d={mult_d:.4f}"
    )
    return lr_g, lr_d, True, epoch_1based, why


# ============================================================
# Train
# ============================================================
def train_gan(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/samples", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.conditional:
        generator = ConditionalGenerator(
            latent_dim=args.latent_dim,
            num_classes=args.num_classes,
            img_channels=1,
            img_size=args.img_size
        ).to(device)

        discriminator = ConditionalDiscriminator(
            num_classes=args.num_classes,
            img_channels=1,
            img_size=args.img_size
        ).to(device)

        print("Using Conditional GAN")
    else:
        generator = Generator(
            latent_dim=args.latent_dim,
            img_channels=1,
            img_size=args.img_size
        ).to(device)

        discriminator = Discriminator(
            img_channels=1,
            img_size=args.img_size
        ).to(device)

        print("Using Standard GAN")


    adversarial_loss = nn.BCEWithLogitsLoss()

    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_g, betas=(args.b1, args.b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(args.b1, args.b2))

    start_epoch = 0

    # Optional adaptive tuner (kept available, but not recommended to mix with auto_lr)
    adaptive_tuner = None
    if args.use_adaptive_tuning:
        adaptive_tuner = AdaptiveHyperparameterTuner(args.lr_g, args.lr_d, args.n_critic, window_size=args.tuning_window)
        print(f"Adaptive tuner enabled (window={args.tuning_window})")

    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)

        generator.load_state_dict(checkpoint["generator_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

        # IMPORTANT: really override LRs from CLI
        force_optimizer_lr(optimizer_G, args.lr_g)
        force_optimizer_lr(optimizer_D, args.lr_d)

        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"].cpu())
        if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu())
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])

        print(f"Resumed epoch {checkpoint['epoch']} -> start at {start_epoch}")
        print(f"   Forced LRs: lr_g={optimizer_G.param_groups[0]['lr']:.8f}, lr_d={optimizer_D.param_groups[0]['lr']:.8f}")

    elif args.pretrained_generator:
        print(f"Loading pretrained generator: {args.pretrained_generator}")
        generator.load_state_dict(torch.load(args.pretrained_generator, map_location=device, weights_only=False))
        print("Generator loaded (D+optimizers fresh)")

    print(f"\nLoading dataset from: {args.data_root}")
    dataloader = Get_dataloader(
        data_root=args.data_root,
        shuffle=True,   
        batch_size=args.batch_size,
        img_size=args.img_size,
        conditional=args.conditional,
        num_workers=args.num_workers
    )
    print(f"Total batches per epoch: {len(dataloader)}")
    print(f"Training epochs: {start_epoch+1} → {args.epochs}")

    # Live plot + history
    fig, ax, line_d, line_g = init_live_plot()
    epochs_x, d_hist, g_hist = [], [], []

    # Rolling mean for auto LR decisions
    ROLL = 5
    d_roll = deque(maxlen=ROLL)
    g_roll = deque(maxlen=ROLL)
    last_adjust_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        g_losses = []
        d_losses = []

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for i, batch_data in enumerate(progress_bar):
            if args.conditional:
                real_images, real_labels = batch_data
                real_images = real_images.to(device)
                real_labels = real_labels.to(device)
            else:
                real_images = batch_data.to(device)

            batch_size = real_images.size(0)
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            z = torch.randn(batch_size, args.latent_dim, device=device)

            if args.conditional:
                gen_labels = torch.randint(0, args.num_classes, (batch_size,), device=device)
                gen_images = generator(z, gen_labels)
                real_validity = discriminator(real_images, real_labels)
                fake_validity = discriminator(gen_images.detach(), gen_labels)
            else:
                gen_images = generator(z)
                real_validity = discriminator(real_images)
                fake_validity = discriminator(gen_images.detach())

            d_real_loss = adversarial_loss(real_validity, valid)
            d_fake_loss = adversarial_loss(fake_validity, fake)
            d_loss = 0.5 * (d_real_loss + d_fake_loss)

            d_loss.backward()

            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.clip_grad)

            optimizer_D.step()
            d_losses.append(float(d_loss.item()))

            # ---------------------
            # Train Generator (every n_critic)
            # ---------------------
            if i % args.n_critic == 0:
                optimizer_G.zero_grad()
                z = torch.randn(batch_size, args.latent_dim, device=device)

                if args.conditional:
                    gen_labels = torch.randint(0, args.num_classes, (batch_size,), device=device)
                    gen_images = generator(z, gen_labels)
                    validity = discriminator(gen_images, gen_labels)
                else:
                    gen_images = generator(z)
                    validity = discriminator(gen_images)

                g_loss = adversarial_loss(validity, valid)
                g_loss.backward()

                if args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clip_grad)

                optimizer_G.step()
                g_losses.append(float(g_loss.item()))
            else:
                # keep tqdm happy even when G not updated this iter
                g_loss = torch.tensor(np.mean(g_losses) if len(g_losses) else 0.0)

            progress_bar.set_postfix({
                "D_loss": f"{d_loss.item():.4f}",
                "G_loss": f"{(g_loss.item() if torch.is_tensor(g_loss) else g_loss):.4f}",
                "lr_g": f"{optimizer_G.param_groups[0]['lr']:.2e}",
                "lr_d": f"{optimizer_D.param_groups[0]['lr']:.2e}",
            })

        # epoch averages
        avg_d_loss = float(np.mean(d_losses)) if len(d_losses) else 0.0
        avg_g_loss = float(np.mean(g_losses)) if len(g_losses) else 0.0

        current_lr_g = optimizer_G.param_groups[0]["lr"]
        current_lr_d = optimizer_D.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"| D={avg_d_loss:.4f} G={avg_g_loss:.4f} "
            f"| lr_g={current_lr_g:.8f} lr_d={current_lr_d:.8f} "
            f"| n_critic={args.n_critic}"
        )

        # live plot update (once)
        epochs_x.append(epoch + 1)
        d_hist.append(avg_d_loss)
        g_hist.append(avg_g_loss)
        update_live_plot(fig, ax, line_d, line_g, epochs_x, d_hist, g_hist)

        # rolling mean for auto lr controller
        d_roll.append(avg_d_loss)
        g_roll.append(avg_g_loss)
        d_avg = float(np.mean(d_roll))
        g_avg = float(np.mean(g_roll))

        # AUTO LR controller (your "apply auto rate")
        if args.use_auto_lr:
            new_lr_g, new_lr_d, did_adjust, last_adjust_epoch, why = auto_lr_controller(
                d_loss_avg=d_avg,
                g_loss_avg=g_avg,
                optimizer_G=optimizer_G,
                optimizer_D=optimizer_D,
                target_d=(args.target_d_lo, args.target_d_hi),
                target_g=(args.target_g_lo, args.target_g_hi),

                cooldown_epochs=args.auto_lr_cooldown,
                interval_epochs=args.auto_lr_interval,

                # IMPORTANT: prevent runaway / starving D
                lr_g_bounds=(args.lr_g_min, args.lr_g_max),
                lr_d_bounds=(args.lr_d_min, args.lr_d_max),

                epoch_1based=epoch + 1,
                last_adjust_epoch=last_adjust_epoch,

                gain=args.auto_lr_gain,
                max_mult_per_update=args.auto_lr_max_up,
                min_mult_per_update=args.auto_lr_max_dn,
                dominance_deadband=args.auto_lr_deadband,
            )

            if did_adjust:
                print(f"    AUTO-LR applied ({why}) -> lr_g={new_lr_g:.8f}, lr_d={new_lr_d:.8f}")
            else:
                print(f"    AUTO-LR not applied ({why})")

        # Optional: adaptive tuner (do NOT mix with auto_lr ideally)
        if adaptive_tuner is not None and (epoch + 1) % args.tuning_interval == 0:
            adaptive_tuner.update(avg_g_loss, avg_d_loss)
            if adaptive_tuner.should_adjust():
                new_lr_g, new_lr_d, new_n_critic, adjusted = adaptive_tuner.adjust_hyperparameters()
                if adjusted:
                    force_optimizer_lr(optimizer_G, new_lr_g)
                    force_optimizer_lr(optimizer_D, new_lr_d)
                    args.n_critic = new_n_critic
                    print("   Adaptive tuner updated hyperparams")

        # save samples
        if (epoch + 1) % args.sample_interval == 0:
            with torch.no_grad():
                z = torch.randn(16, args.latent_dim, device=device)
                if args.conditional:
                    sample_labels = torch.arange(0, args.num_classes, device=device).repeat(16 // args.num_classes + 1)[:16]
                    gen_images = generator(z, sample_labels)
                else:
                    gen_images = generator(z)

                save_image(gen_images.data, f"{args.output_dir}/samples/epoch_{epoch+1}.png",
                           nrow=4, normalize=True)

        # checkpoints
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_dict = {
                "epoch": epoch,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_G_state_dict": optimizer_G.state_dict(),
                "optimizer_D_state_dict": optimizer_D.state_dict(),
                "g_loss": avg_g_loss,
                "d_loss": avg_d_loss,
                "rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
            }
            if torch.cuda.is_available():
                checkpoint_dict["cuda_rng_state"] = torch.cuda.get_rng_state()

            if adaptive_tuner is not None:
                checkpoint_dict["adaptive_tuner"] = {
                    "lr_g": adaptive_tuner.lr_g,
                    "lr_d": adaptive_tuner.lr_d,
                    "n_critic": adaptive_tuner.n_critic,
                    "g_loss_history": list(adaptive_tuner.g_loss_history),
                    "d_loss_history": list(adaptive_tuner.d_loss_history),
                    "epochs_since_adjustment": adaptive_tuner.epochs_since_adjustment,
                    "adjustment_count": adaptive_tuner.adjustment_count,
                    "oscillation_count": adaptive_tuner.oscillation_count,
                }

            out_path = f"{args.checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth"
            torch.save(checkpoint_dict, out_path)
            print(f" Saved checkpoint: {out_path}")

    torch.save(generator.state_dict(), f"{args.checkpoint_dir}/generator_final.pth")
    torch.save(discriminator.state_dict(), f"{args.checkpoint_dir}/discriminator_final.pth")
    print(f"\n Training complete! Models saved to {args.checkpoint_dir}")


def generate_images(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.conditional:
        generator = ConditionalGenerator(args.latent_dim, args.num_classes, args.img_size).to(device)
    else:
        generator = Generator(args.latent_dim, args.img_size).to(device)

    generator.load_state_dict(torch.load(args.generator_path, map_location=device, weights_only=False))
    generator.eval()

    print(f"Generating {args.num_images} synthetic X-ray images...")
    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for i in tqdm(range(args.num_images)):
            z = torch.randn(1, args.latent_dim, device=device)
            if args.conditional:
                label = torch.tensor([args.class_label], device=device)
                gen_image = generator(z, label)
            else:
                gen_image = generator(z)

            save_image(gen_image, f"{args.output_dir}/synthetic_xray_{i+1}.png", normalize=True)

    print(f" Generated {args.num_images} images in {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN for Synthetic X-ray Generation")

    parser.add_argument("--mode", type=str, default="train", choices=["train", "generate"])
    parser.add_argument("--pretrained_generator", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)

    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--num_classes", type=int, default=3)

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_g", type=float, default=0.0002)
    parser.add_argument("--lr_d", type=float, default=0.0002)
    parser.add_argument("--b1", type=float, default=0.5)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_critic", type=int, default=1)
    parser.add_argument("--clip_grad", type=float, default=None)

    # Adaptive tuner
    parser.add_argument("--use_adaptive_tuning", action="store_true")
    parser.add_argument("--tuning_window", type=int, default=10)
    parser.add_argument("--tuning_interval", type=int, default=1)

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--checkpoint_dir", type=str, default="./models/generator/checkpoints")
    parser.add_argument("--sample_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=50)

    # Generate
    parser.add_argument("--generator_path", type=str, default="./models/generator/generator_final.pth")
    parser.add_argument("--num_images", type=int, default=1000)
    parser.add_argument("--class_label", type=int, default=0)

    #  AUTO LR controller flags - IMPROVED DEFAULTS
    parser.add_argument("--use_auto_lr", action="store_true", help="Enable auto LR controller")
    parser.add_argument("--auto_lr_interval", type=int, default=5, help="Apply AUTO-LR every N epochs")
    parser.add_argument("--auto_lr_gain", type=float, default=0.40, help="Reduced from 0.55")
    parser.add_argument("--auto_lr_max_up", type=float, default=1.02, help="Max 4% increase per update")
    parser.add_argument("--auto_lr_max_dn", type=float, default=0.97, help="Max 4% decrease per update")
    parser.add_argument("--auto_lr_deadband", type=float, default=0.12, help="Increased deadband")
    parser.add_argument("--auto_lr_cooldown", type=int, default=1, help="Epoch cooldown between updates")

    parser.add_argument("--target_d_lo", type=float, default=0.3)
    parser.add_argument("--target_d_hi", type=float, default=0.6)
    parser.add_argument("--target_g_lo", type=float, default=1.3)
    parser.add_argument("--target_g_hi", type=float, default=1.8)
    parser.add_argument("--lr_g_min", type=float, default=0.00008)
    parser.add_argument("--lr_g_max", type=float, default=0.005)
    parser.add_argument("--lr_d_min", type=float, default=0.000003)  # Raised floor for D
    parser.add_argument("--lr_d_max", type=float, default=0.0002)

    args = parser.parse_args()

    if args.mode == "train":
        train_gan(args)
    else:
        generate_images(args)
"""
RUN THIS TO TRAIN THE MODEL

test: python notebooks/GanTraining/modeltraining.py --mode train --data_root data/interim/GAN/Stage1/color_clahe_1500x1000_noborder_aug/part1 --epochs 400 --batch_size 8 --img_size 256 --lr_g 0.0002 --lr_d 0.00001 --use_auto_lr --auto_lr_cooldown 1 --auto_lr_interval 2 --auto_lr_gain 0.40 --auto_lr_max_up 1.10 --auto_lr_max_dn 0.90 -
-auto_lr_deadband 0.12 --lr_d_min 0.000001 --clip_grad 1.0 --output_dir ./mode
ls/generator/outputs --checkpoint_dir ./models/generator/checkpoints

WITH ADAPTIVE BAYESIAN OPTIMIZATION (Recommended - Auto-balances D and G) lr_g -> 0.0002 lr_d -> 0.000003

WITHOUT PRETRAIN MODEL (Standard Training)
# example after you change argparse + controller
python notebooks/GanTraining/modeltraining.py --mode train --data_root data/interim/GAN/Stage1/color_clahe_1500x1000_noborder_aug/part1 --epochs 400 --batch_size 8 --img_size 256 --lr_g 0.0002 --lr_d 0.00001 --use_auto_lr --auto_lr_cooldown 1 --auto_lr_interval 5 --auto_lr_gain 0.40 --auto_lr_max_up 1.04 --auto_lr_max_dn 0.96 --auto_lr_deadband 0.12 --lr_d_min 0.000003 --clip_grad 1.0 --output_dir ./models/generator/outputs --checkpoint_dir ./models/generator/checkpoints



WITH PRETRAIN MODEL/CHECKPOINT 
python notebooks/GanTraining/modeltraining.py --mode train --resume_from ./models/generator/checkpoints/checkpoint_epoch_100.pth --data_root data/interim/Stage1/color_clahe_1500x1000_noborder_aug --epochs 200 --batch_size 17 --img_size 256 --lr_g 0.0002 --lr_d 0.00001 --use_adaptive_tuning --tuning_window 5 --tuning_interval 1 --clip_grad 1.0 --output_dir ./models/generator/outputs --checkpoint_dir ./models/generator/checkpoints






WITH PRETRAIN MODEL/CHECKPOINT (Manual Hyperparameters)
python notebooks/GanTraining/modeltraining.py 
  --mode train 
  --resume_from ./models/generator/checkpoints/checkpointV1.pth 
  --data_root data/interim/Stage1/color_clahe_1500x1000_noborder_aug 
  --epochs 200 
  --batch_size 17 
  --img_size 256 
  --lr_g 0.0002 
  --lr_d 0.000015 
  --output_dir ./models/generator/outputs 
  --checkpoint_dir ./models/generator/checkpoints

"""