# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Adaptive Learning Rate Controller for StyleGAN2-ADA

"""
Adaptive LR controller that maintains:
- D loss: 0.3 - 0.7
- G loss: 1.3 - 2.5
"""

import numpy as np
from collections import deque

class StyleGAN2AdaptiveLR:
    def __init__(self, initial_lr_g=0.0025, initial_lr_d=0.005, 
                 target_d_range=(0.3, 0.7), target_g_range=(1.5, 2.0),
                 window_size=2, adjustment_interval=2, cooldown_ticks=1,
                 lr_bounds_d=(0.000001, 0.01), lr_bounds_g=(0.0025, 0.08)):
        self.lr_g = initial_lr_g
        self.lr_d = initial_lr_d
        self.target_d_min, self.target_d_max = target_d_range
        self.target_g_min, self.target_g_max = target_g_range
        self.lr_g_min, self.lr_g_max = lr_bounds_g
        self.lr_d_min, self.lr_d_max = lr_bounds_d
        self.window_size = window_size
        self.adjustment_interval = adjustment_interval
        self.cooldown_ticks = cooldown_ticks
        self.d_loss_history = deque(maxlen=window_size)
        self.g_loss_history = deque(maxlen=window_size)
        self.ticks_since_adjustment = 99
        self.total_ticks = 0
        self.num_adjustments = 0
        
    def update(self, d_loss, g_loss):
        self.d_loss_history.append(float(d_loss))
        self.g_loss_history.append(float(g_loss))
        self.ticks_since_adjustment += 1
        self.total_ticks += 1
        
    def should_adjust(self):
        if len(self.d_loss_history) < self.window_size:
            return False
        if self.total_ticks % self.adjustment_interval != 0:
            return False
        if self.ticks_since_adjustment < self.cooldown_ticks:
            return False
        return True
        
    def adjust_learning_rates(self, optimizer_G, optimizer_D, verbose=True):
        if not self.should_adjust():
            return self.lr_g, self.lr_d, False, "cooldown"
            
        avg_d = np.mean(self.d_loss_history)
        avg_g = np.mean(self.g_loss_history)
        
        mult_g = 1.0
        mult_d = 1.0
        
        # D dominating
        if avg_d < self.target_d_min and avg_g > self.target_g_max:
            strength = min((self.target_d_min - avg_d) / self.target_d_min + 
                          (avg_g - self.target_g_max) / self.target_g_max, 0.5)
            mult_d = 1.0 - (strength * 0.5)
            mult_g = 1.0 + (strength * 0.8)
            reason = f"D_dominating [D={avg_d:.3f} G={avg_g:.3f}]"
            
        # G dominating
        elif avg_d > self.target_d_max and avg_g < self.target_g_min:
            strength = min((avg_d - self.target_d_max) / self.target_d_max + 
                          (self.target_g_min - avg_g) / self.target_g_min, 0.4)
            mult_d = 1.0 + (strength * 0.6)
            mult_g = 1.0 - (strength * 0.4)
            reason = f"G_dominating [D={avg_d:.3f} G={avg_g:.3f}]"
            
        # D too strong
        elif avg_d < self.target_d_min:
            strength = min((self.target_d_min - avg_d) / self.target_d_min, 0.3)
            mult_d = 1.0 - (strength * 0.4)
            mult_g = 1.0 + (strength * 0.2)
            reason = f"D_too_strong [D={avg_d:.3f}]"
            
        # D too weak
        elif avg_d > self.target_d_max:
            strength = min((avg_d - self.target_d_max) / self.target_d_max, 0.3)
            mult_d = 1.0 + (strength * 0.5)
            mult_g = 1.0 - (strength * 0.2)
            reason = f"D_too_weak [D={avg_d:.3f}]"
            
        # G too weak
        elif avg_g > self.target_g_max:
            strength = min((avg_g - self.target_g_max) / self.target_g_max, 0.3)
            mult_g = 1.0 + (strength * 0.5)
            mult_d = 1.0 - (strength * 0.2)
            reason = f"G_too_weak [G={avg_g:.3f}]"
            
        # G too strong
        elif avg_g < self.target_g_min:
            strength = min((self.target_g_min - avg_g) / self.target_g_min, 0.3)
            mult_g = 1.0 - (strength * 0.4)
            mult_d = 1.0 + (strength * 0.2)
            reason = f"G_too_strong [G={avg_g:.3f}]"
        else:
            return self.lr_g, self.lr_d, False, "in_range"
            
        # Cap multipliers
        mult_g = max(0.85, min(1.20, mult_g))
        mult_d = max(0.85, min(1.20, mult_d))
        
        # Apply
        old_lr_g = self.lr_g
        old_lr_d = self.lr_d
        self.lr_g = max(self.lr_g_min, min(self.lr_g_max, self.lr_g * mult_g))
        self.lr_d = max(self.lr_d_min, min(self.lr_d_max, self.lr_d * mult_d))
        
        for pg in optimizer_G.param_groups:
            pg['lr'] = self.lr_g
        for pg in optimizer_D.param_groups:
            pg['lr'] = self.lr_d
            
        self.ticks_since_adjustment = 0
        self.num_adjustments += 1
        
        if verbose:
            print(f"\n🔧 LR Adjustment #{self.num_adjustments}: {reason}")
            print(f"   G: {old_lr_g:.6f} → {self.lr_g:.6f} | D: {old_lr_d:.6f} → {self.lr_d:.6f}\n")
            
        return self.lr_g, self.lr_d, True, reason
