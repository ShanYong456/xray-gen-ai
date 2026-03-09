import torch
import torch.nn.functional as F

from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    """pix2pix model with OPTIONAL physics-style delta compositing on an empty tray E.

    Input:  A (mask-like condition) + E (empty tray)
    Target: B (real X-ray)

    Adds detail supervision so the model learns realism from real_B:
      - delta supervision: delta_gt = OD(B) - OD(E)
      - gradient loss on fake_B vs real_B (inside object region)
      - gradient loss on delta_od vs delta_gt (inside object region)
      - OPTIONAL: Laplacian loss on fake_B vs real_B (very good for internal details)
      - OPTIONAL: SSIM loss inside object region
      - OPTIONAL: Region stats loss (mean/std) inside object region
      - tray mask T to restrict where changes are allowed (IMPORTANT for tray alignment)

    NOTE:
    To prevent objects appearing outside the tray, we:
      1) Clamp the object mask M := M * T
      2) Clamp the predicted delta_od := delta_od * T
      3) Ensure ALL losses that use object region use (M * T) consistently

    This requires your dataset to provide input['T'] (1 inside tray, 0 outside),
    derived from your "empty tray mask png" and aligned to E/A/B resolution.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm="batch", netG="unet_256", dataset_mode="aligned")
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="vanilla")
            parser.add_argument("--lambda_L1", type=float, default=100.0, help="weight for L1 loss")

        # ---- Physics / delta-compositing options ----
        parser.add_argument("--compose_eps", type=float, default=1e-6,
                            help="epsilon for log/exp stability in delta composition")
        parser.add_argument("--delta_scale", type=float, default=1.0,
                            help="scale applied to predicted delta before adding in OD-space")

        parser.add_argument("--delta_positive", action="store_true",
                            help="force delta_od >= 0 using softplus (recommended for X-ray attenuation).")
        parser.add_argument("--delta_max", type=float, default=6.0,
                            help="Clamp OD delta to [0, delta_max] to prevent over-darkening/explosions.")

        parser.add_argument("--od_gamma", type=float, default=1.0,
                            help="Gamma applied before OD: OD = -log(I^gamma + eps). <1 softens OD.")

        # Mask handling
        parser.add_argument("--mask_nc", type=int, default=3,
                            help="How many channels in A correspond to the semantic mask (before E). "
                                 "For one-hot shampoo+blade use 2; for RGB palette mask use 3.")
        parser.add_argument("--mask_thr", type=float, default=0.05,
                            help="threshold in [0,1] after unnormalizing A to detect object pixels")

        # ---- NEW: optional thickness proxy channel support ----
        parser.add_argument("--use_thickness_channel", action="store_true",
                            help="If set, A is expected to include extra thickness proxy channel(s) "
                                 "(e.g., distance transform) in addition to mask channels.")
        parser.add_argument("--thickness_nc", type=int, default=1,
                            help="Number of thickness channels appended in A (typically 1). "
                                 "Used only for documentation/validation; mask extraction still uses mask_nc.")

        # Masked L1 option (object + background identity)
        parser.add_argument("--use_masked_l1", action="store_true",
                            help="use masked object L1 + background identity L1 (requires use_delta_comp).")
        parser.add_argument("--lambda_bg", type=float, default=1.5,
                            help="background identity strength vs empty tray outside mask")

        # Regularize delta outside mask to be ~0
        parser.add_argument("--lambda_delta_bg", type=float, default=5.0,
                            help="Penalty weight for delta outside object mask (only used with use_delta_comp).")

        # ---- Direct delta supervision (OD-space) ----
        parser.add_argument("--use_delta_supervision", action="store_true",
                            help="supervise delta_od with delta_gt = OD(B)-OD(E) inside mask.")
        parser.add_argument("--lambda_delta", type=float, default=50.0,
                            help="Weight for delta supervision loss (only used with use_delta_supervision).")

        # ---- Tray constraint (REQUIRED for your 'empty tray mask png' restriction) ----
        parser.add_argument("--use_tray_mask", action="store_true",
                            help="expect input['T'] tray mask (1 inside tray, 0 outside) and constrain composition/loss.")

        # ---- Detail supervision using real_B (edge/gradient losses) ----
        parser.add_argument("--use_grad_loss", action="store_true",
                            help="Sobel gradient loss on fake_B vs real_B inside object region.")
        parser.add_argument("--lambda_grad", type=float, default=10.0,
                            help="Weight for gradient loss on fake_B vs real_B (try 5~30).")

        parser.add_argument("--use_delta_grad_loss", action="store_true",
                            help="Sobel gradient loss on delta_od vs delta_gt inside object region.")
        parser.add_argument("--lambda_delta_grad", type=float, default=10.0,
                            help="Weight for gradient loss on delta_od vs delta_gt (try 5~30).")

        # ---- Laplacian loss ----
        parser.add_argument("--use_lap_loss", action="store_true",
                            help="Laplacian loss on fake_B vs real_B inside object region (very effective for internal structure).")
        parser.add_argument("--lambda_lap", type=float, default=10.0,
                            help="Weight for Laplacian loss (try 5~30).")

        # ---- SSIM loss ----
        parser.add_argument("--use_ssim_loss", action="store_true",
                            help="SSIM loss on fake_B vs real_B inside object region (stabilizes and improves structure).")
        parser.add_argument("--lambda_ssim", type=float, default=5.0,
                            help="Weight for SSIM loss (try 1~10).")

        # ---- Region stats loss ----
        parser.add_argument("--use_region_stats", action="store_true",
                            help="Match mean/std of fake_B to real_B inside object region (helps attenuation realism).")
        parser.add_argument("--lambda_stats", type=float, default=5.0,
                            help="Weight for region stats loss (try 1~10).")

        # ---- Soft mask options (helps remove mask artifacts) ----
        parser.add_argument("--use_soft_mask", action="store_true",
                            help="Use soft mask instead of binary mask.")
        parser.add_argument("--mask_soft_beta", type=float, default=30.0,
                            help="Soft mask sharpness. Larger = sharper mask edge.")

        parser.add_argument("--mask_blur_ksize", type=int, default=0,
                            help="Gaussian blur kernel size for mask smoothing (0 disables).")

        parser.add_argument("--mask_blur_sigma", type=float, default=1.2,
                            help="Gaussian blur sigma for mask smoothing.")

        parser.add_argument("--mask_noise_std", type=float, default=0.0,
                            help="Small noise added to mask channels during training (e.g., 0.02).")
        
                # ---- Synthetic-only training support ----
        parser.add_argument("--lambda_syn_tv", type=float, default=1.0,
                            help="TV smoothness on delta_od for synthetic-only batches.")
        parser.add_argument("--lambda_syn_mag", type=float, default=0.2,
                            help="Weak magnitude regularization on delta_od for synthetic-only batches.")
        parser.add_argument("--lambda_syn_mask_mean", type=float, default=0.0,
                            help="Optional weak encouragement for non-zero response inside object mask.")

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = [
            "G_GAN", "G_L1",
            "G_delta_bg", "G_delta",
            "G_grad", "G_delta_grad",
            "G_lap", "G_ssim", "G_stats",
            "G_syn_tv", "G_syn_mag", "G_syn_mask_mean",
            "D_real", "D_fake"
        ]
        self.visual_names = ["real_A", "fake_B", "real_B"]

        self.model_names = ["G", "D"] if self.isTrain else ["G"]
        self.device = opt.device

        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
            not opt.no_dropout, opt.init_type, opt.init_gain
        )

        if self.isTrain:
            self.netD = networks.define_D(
                opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain
            )
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers += [self.optimizer_G, self.optimizer_D]

        # holders
        self.empty_E = None
        self.delta_raw = None
        self.delta_od = None
        self.delta_gt = None
        self.mask_M = None
        self.tray_T = None

        # init loss holders
        z = torch.tensor(0.0, device=self.device)
        self.loss_G_delta_bg = z.clone()
        self.loss_G_delta = z.clone()
        self.loss_G_grad = z.clone()
        self.loss_G_delta_grad = z.clone()
        self.loss_G_lap = z.clone()
        self.loss_G_ssim = z.clone()
        self.loss_G_stats = z.clone()
        self.loss_G_syn_tv = z.clone()
        self.loss_G_syn_mag = z.clone()
        self.loss_G_syn_mask_mean = z.clone()

        self._init_kernels()

    def _init_kernels(self):
        # Sobel kernels (3x3)
        kx = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3) / 4.0
        ky = torch.tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3) / 4.0

        # Laplacian kernel (3x3)
        kl = torch.tensor([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)

        self.sobel_kx = kx.to(self.device)
        self.sobel_ky = ky.to(self.device)
        self.lap_k = kl.to(self.device)

    def _gaussian_blur(self, x, ksize=5, sigma=1.2):
        if ksize <= 1:
            return x
        if ksize % 2 == 0:
            ksize += 1

        C = x.shape[1]
        grid = torch.arange(ksize, device=x.device) - (ksize - 1) / 2
        gaussian = torch.exp(-(grid ** 2) / (2 * sigma ** 2))
        gaussian = gaussian / gaussian.sum()

        kernel_x = gaussian.view(1, 1, 1, ksize).repeat(C, 1, 1, 1)
        kernel_y = gaussian.view(1, 1, ksize, 1).repeat(C, 1, 1, 1)

        x = F.conv2d(x, kernel_x, padding=(0, ksize // 2), groups=C)
        x = F.conv2d(x, kernel_y, padding=(ksize // 2, 0), groups=C)
        return x

    def set_input(self, input):
        AtoB = self.opt.direction == "AtoB"

        A = input["A" if AtoB else "B"].to(self.device)

        # add small noise to mask channels to break grid artifacts
        if self.isTrain:
            noise_std = float(getattr(self.opt, "mask_noise_std", 0.0))
            if noise_std > 0:
                mask_nc = int(getattr(self.opt, "mask_nc", 3))
                noise = torch.randn_like(A[:, :mask_nc]) * noise_std
                A[:, :mask_nc] = torch.clamp(A[:, :mask_nc] + noise, -1.0, 1.0)

        # synthetic flag
        is_synth = input.get("is_synthetic", False)
        if isinstance(is_synth, torch.Tensor):
            is_synth = bool(is_synth.flatten()[0].item())
        self.is_synthetic = bool(is_synth)

        # real target may be missing for synthetic batches
        b_key = "B" if AtoB else "A"
        if b_key in input and input[b_key] is not None:
            self.real_B = input[b_key].to(self.device)
            self.has_real_B = True
        else:
            self.real_B = None
            self.has_real_B = False

        if getattr(self.opt, "use_delta_comp", False):
            if "E" not in input:
                raise KeyError("use_delta_comp enabled but input has no key 'E'.")
            E = input["E"].to(self.device)
            self.empty_E = E
            self.real_A = torch.cat([A, E], dim=1)
        else:
            self.real_A = A
            self.empty_E = None

        # tray mask
        if getattr(self.opt, "use_tray_mask", False):
            if "T" not in input:
                raise KeyError("use_tray_mask set but input has no key 'T' (tray mask).")
            self.tray_T = input["T"].to(self.device)
        else:
            self.tray_T = None

        self.image_paths = input.get("A_paths" if AtoB else "B_paths", [])

    def _build_object_mask_from_A(self):
        """Build object mask from first mask_nc channels. Supports soft masks + blur."""
        mask_nc = int(getattr(self.opt, "mask_nc", 3))
        A_only = self.real_A[:, :mask_nc, :, :]   # [-1,1]
        A01 = (A_only + 1.0) * 0.5                # [0,1]

        thr = float(getattr(self.opt, "mask_thr", 0.05))
        strength = A01.sum(dim=1, keepdim=True)

        if getattr(self.opt, "use_soft_mask", False):
            beta = float(getattr(self.opt, "mask_soft_beta", 30.0))
            thr_scaled = thr * mask_nc
            M = torch.sigmoid(beta * (strength - thr_scaled))
        else:
            M = (strength > thr).float()

        blur_k = int(getattr(self.opt, "mask_blur_ksize", 0))
        blur_sigma = float(getattr(self.opt, "mask_blur_sigma", 1.2))
        if blur_k > 1:
            M = self._gaussian_blur(M, blur_k, blur_sigma)

        M = torch.clamp(M, 0.0, 1.0)
        return A_only, M

    def _expand_like(self, mask_1ch: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if mask_1ch is None:
            return None
        if mask_1ch.shape[1] == x.shape[1]:
            return mask_1ch
        if mask_1ch.shape[1] == 1 and x.shape[1] != 1:
            return mask_1ch.expand(-1, x.shape[1], -1, -1)
        return mask_1ch

    def _sobel_mag(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        kx = self.sobel_kx.to(dtype=x.dtype).repeat(C, 1, 1, 1)
        ky = self.sobel_ky.to(dtype=x.dtype).repeat(C, 1, 1, 1)
        gx = F.conv2d(x, kx, padding=1, groups=C)
        gy = F.conv2d(x, ky, padding=1, groups=C)
        return torch.sqrt(gx * gx + gy * gy + 1e-12)

    def _laplacian(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        k = self.lap_k.to(dtype=x.dtype).repeat(C, 1, 1, 1)
        return F.conv2d(x, k, padding=1, groups=C)

    def _ssim_map(self, x: torch.Tensor, y: torch.Tensor, window_size: int = 7) -> torch.Tensor:
        pad = window_size // 2
        mu_x = F.avg_pool2d(x, window_size, stride=1, padding=pad)
        mu_y = F.avg_pool2d(y, window_size, stride=1, padding=pad)

        sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=pad) - mu_x * mu_x
        sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=pad) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=pad) - mu_x * mu_y

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
            (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2) + 1e-12
        )
        return ssim

    def forward(self):
        if not getattr(self.opt, "use_delta_comp", False):
            self.fake_B = self.netG(self.real_A)
            self.delta_raw = None
            self.delta_od = None
            self.delta_gt = None
            self.mask_M = None
            return

        # predict delta in OD-space (raw)
        self.delta_raw = self.netG(self.real_A)

        # build object mask from A, then HARD restrict to tray region
        _, M = self._build_object_mask_from_A()
        if getattr(self.opt, "use_tray_mask", False) and self.tray_T is not None:
            M = M * self.tray_T
        self.mask_M = M

        eps = float(getattr(self.opt, "compose_eps", 1e-6))
        delta_scale = float(getattr(self.opt, "delta_scale", 1.0))
        delta_max = float(getattr(self.opt, "delta_max", 6.0))
        delta_positive = bool(getattr(self.opt, "delta_positive", False))

        # empty tray intensity -> OD
        E01 = torch.clamp((self.empty_E + 1.0) * 0.5, 0.0, 1.0)
        od_gamma = float(getattr(self.opt, "od_gamma", 1.0))
        if od_gamma != 1.0:
            E01 = torch.pow(E01, od_gamma)
        OD_E = -torch.log(E01 + eps)

        # delta constraint
        if delta_positive:
            delta_od = F.softplus(self.delta_raw)
        else:
            delta_od = self.delta_raw

        delta_od = delta_od * delta_scale
        if delta_positive:
            delta_od = torch.clamp(delta_od, 0.0, delta_max)

        # VERY IMPORTANT: enforce tray constraint on delta itself
        if getattr(self.opt, "use_tray_mask", False) and self.tray_T is not None:
            delta_od = delta_od * self._expand_like(self.tray_T, delta_od)

        self.delta_od = delta_od

        # tray mask used for composition
        T = self.tray_T if (getattr(self.opt, "use_tray_mask", False) and self.tray_T is not None) else torch.ones_like(M)
        M_exp = self._expand_like(M, delta_od)
        T_exp = self._expand_like(T, delta_od)

        # compose ONLY inside tray & object region
        OD_pred = OD_E + (T_exp * M_exp) * delta_od

        I_pred = torch.exp(-OD_pred)
        I_pred = torch.clamp(I_pred, 0.0, 1.0)

        # outside tray: force back to empty tray
        if getattr(self.opt, "use_tray_mask", False) and self.tray_T is not None:
            I_pred = T_exp * I_pred + (1.0 - T_exp) * E01

        self.fake_B = I_pred * 2.0 - 1.0

        # delta_gt for supervision (OD-space)
        if getattr(self.opt, "use_delta_supervision", False) and self.has_real_B:
            B01 = torch.clamp((self.real_B + 1.0) * 0.5, 0.0, 1.0)
            if od_gamma != 1.0:
                B01 = torch.pow(B01, od_gamma)
            OD_B = -torch.log(B01 + eps)
            delta_gt = OD_B - OD_E
            if delta_positive:
                delta_gt = torch.clamp(delta_gt, 0.0, delta_max)

            if getattr(self.opt, "use_tray_mask", False) and self.tray_T is not None:
                delta_gt = delta_gt * self._expand_like(self.tray_T, delta_gt)

            self.delta_gt = delta_gt
        else:
            self.delta_gt = None
    
    def _tv_loss(self, x: torch.Tensor) -> torch.Tensor:
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        return dx + dy

    def backward_D(self):
        if self.is_synthetic or (not self.has_real_B):
            z = torch.tensor(0.0, device=self.device)
            self.loss_D_fake = z.clone()
            self.loss_D_real = z.clone()
            self.loss_D = z.clone()
            return

        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        z = torch.tensor(0.0, device=self.device)

        if (not self.is_synthetic) and self.has_real_B:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        else:
            self.loss_G_GAN = z.clone()

        z = torch.tensor(0.0, device=self.device)
        self.loss_G_delta_bg = z.clone()
        self.loss_G_delta = z.clone()
        self.loss_G_grad = z.clone()
        self.loss_G_delta_grad = z.clone()
        self.loss_G_lap = z.clone()
        self.loss_G_ssim = z.clone()
        self.loss_G_stats = z.clone()
        self.loss_G_syn_tv = z.clone()
        self.loss_G_syn_mag = z.clone()
        self.loss_G_syn_mask_mean = z.clone()

        # build object region, then restrict to tray
        _, M_obj_1 = self._build_object_mask_from_A()
        if getattr(self.opt, "use_tray_mask", False) and self.tray_T is not None:
            M_obj_1 = M_obj_1 * self.tray_T

        T_1 = self.tray_T if (getattr(self.opt, "use_tray_mask", False) and self.tray_T is not None) else torch.ones_like(M_obj_1)

        M_obj = self._expand_like(M_obj_1, self.fake_B)
        T = self._expand_like(T_1, self.fake_B)

        # ---- L1 loss ----
                # ---- L1 loss ----
        if self.has_real_B:
            if getattr(self.opt, "use_delta_comp", False) and getattr(self.opt, "use_masked_l1", False):
                eps = 1e-6
                obj_region = M_obj * T
                bg_region = (1.0 - M_obj) * T

                obj_num = torch.sum(torch.abs(self.fake_B - self.real_B) * obj_region)
                obj_den = torch.sum(obj_region) + eps
                obj_l1 = obj_num / obj_den

                bg_num = torch.sum(torch.abs(self.fake_B - self.empty_E) * bg_region)
                bg_den = torch.sum(bg_region) + eps
                bg_l1 = bg_num / bg_den

                lambda_bg = float(getattr(self.opt, "lambda_bg", 1.5))
                self.loss_G_L1 = (obj_l1 + lambda_bg * bg_l1) * self.opt.lambda_L1

                lam_delta_bg = float(getattr(self.opt, "lambda_delta_bg", 5.0))
                if lam_delta_bg > 0.0 and self.delta_od is not None and self.mask_M is not None:
                    Mch = self._expand_like(self.mask_M, self.delta_od)
                    Tch = self._expand_like(T_1, self.delta_od)
                    inside_bg = (1.0 - Mch) * Tch
                    self.loss_G_delta_bg = torch.mean(torch.abs(self.delta_od) * inside_bg) * lam_delta_bg
            else:
                if getattr(self.opt, "use_tray_mask", False) and self.tray_T is not None:
                    eps = 1e-6
                    num = torch.sum(torch.abs(self.fake_B - self.real_B) * T)
                    den = torch.sum(T) + eps
                    self.loss_G_L1 = (num / den) * self.opt.lambda_L1
                else:
                    self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = z.clone()

            lam_delta_bg = float(getattr(self.opt, "lambda_delta_bg", 5.0))
            if lam_delta_bg > 0.0 and self.delta_od is not None and self.mask_M is not None:
                Mch = self._expand_like(self.mask_M, self.delta_od)
                Tch = self._expand_like(T_1, self.delta_od)
                inside_bg = (1.0 - Mch) * Tch
                self.loss_G_delta_bg = torch.mean(torch.abs(self.delta_od) * inside_bg) * lam_delta_bg

        # ---- delta supervision loss (OD) ----
        if getattr(self.opt, "use_delta_comp", False) and getattr(self.opt, "use_delta_supervision", False):
            if self.delta_gt is not None and self.delta_od is not None:
                lam_delta = float(getattr(self.opt, "lambda_delta", 50.0))
                eps = 1e-6
                Mch = self._expand_like(M_obj_1, self.delta_od)
                Tch = self._expand_like(T_1, self.delta_od)
                region = Mch * Tch
                num = torch.sum(torch.abs(self.delta_od - self.delta_gt) * region)
                den = torch.sum(region) + eps
                self.loss_G_delta = lam_delta * (num / den)

        # ---- gradient loss on image ----
        if self.has_real_B and getattr(self.opt, "use_grad_loss", False):
            lam_grad = float(getattr(self.opt, "lambda_grad", 10.0))
            eps = 1e-6
            g_fake = self._sobel_mag(self.fake_B)
            g_real = self._sobel_mag(self.real_B)
            region = M_obj * T
            num = torch.sum(torch.abs(g_fake - g_real) * region)
            den = torch.sum(region) + eps
            self.loss_G_grad = lam_grad * (num / den)

        # ---- gradient loss on delta ----
        if getattr(self.opt, "use_delta_grad_loss", False):
            if self.delta_gt is not None and self.delta_od is not None:
                lam_dg = float(getattr(self.opt, "lambda_delta_grad", 10.0))
                eps = 1e-6
                g_d = self._sobel_mag(self.delta_od)
                g_gt = self._sobel_mag(self.delta_gt)
                region = self._expand_like(M_obj_1, self.delta_od) * self._expand_like(T_1, self.delta_od)
                num = torch.sum(torch.abs(g_d - g_gt) * region)
                den = torch.sum(region) + eps
                self.loss_G_delta_grad = lam_dg * (num / den)

        # ---- Laplacian loss on image ----
        if self.has_real_B and getattr(self.opt, "use_lap_loss", False):
            lam_lap = float(getattr(self.opt, "lambda_lap", 10.0))
            eps = 1e-6
            lap_fake = self._laplacian(self.fake_B)
            lap_real = self._laplacian(self.real_B)
            region = M_obj * T
            num = torch.sum(torch.abs(lap_fake - lap_real) * region)
            den = torch.sum(region) + eps
            self.loss_G_lap = lam_lap * (num / den)

        # ---- SSIM loss ----
        if self.has_real_B and getattr(self.opt, "use_ssim_loss", False):
            lam_ssim = float(getattr(self.opt, "lambda_ssim", 5.0))
            eps = 1e-6
            f01 = torch.clamp((self.fake_B + 1.0) * 0.5, 0.0, 1.0)
            r01 = torch.clamp((self.real_B + 1.0) * 0.5, 0.0, 1.0)
            ssim_map = self._ssim_map(f01, r01)
            region = M_obj * T
            num = torch.sum((1.0 - ssim_map) * region)
            den = torch.sum(region) + eps
            self.loss_G_ssim = lam_ssim * (num / den)

        # ---- Region stats loss ----
        if self.has_real_B and getattr(self.opt, "use_region_stats", False):
            lam_stats = float(getattr(self.opt, "lambda_stats", 5.0))
            eps = 1e-6
            f01 = torch.clamp((self.fake_B + 1.0) * 0.5, 0.0, 1.0)
            r01 = torch.clamp((self.real_B + 1.0) * 0.5, 0.0, 1.0)
            region = M_obj * T

            reg_sum = torch.sum(region, dim=(2, 3), keepdim=True) + eps
            f_mean = torch.sum(f01 * region, dim=(2, 3), keepdim=True) / reg_sum
            r_mean = torch.sum(r01 * region, dim=(2, 3), keepdim=True) / reg_sum

            f_var = torch.sum(((f01 - f_mean) ** 2) * region, dim=(2, 3), keepdim=True) / reg_sum
            r_var = torch.sum(((r01 - r_mean) ** 2) * region, dim=(2, 3), keepdim=True) / reg_sum
            f_std = torch.sqrt(f_var + 1e-12)
            r_std = torch.sqrt(r_var + 1e-12)

            self.loss_G_stats = lam_stats * (
                torch.mean(torch.abs(f_mean - r_mean)) + torch.mean(torch.abs(f_std - r_std))
            )
        
        # ---- Synthetic-only regularization ----
        if self.is_synthetic or (not self.has_real_B):
            if self.delta_od is not None:
                lam_tv = float(getattr(self.opt, "lambda_syn_tv", 1.0))
                lam_mag = float(getattr(self.opt, "lambda_syn_mag", 0.2))
                lam_mask_mean = float(getattr(self.opt, "lambda_syn_mask_mean", 0.0))

                Mch = self._expand_like(M_obj_1, self.delta_od)
                Tch = self._expand_like(T_1, self.delta_od)
                region = Mch * Tch
                eps = 1e-6

                # smooth but not noisy inside object region
                self.loss_G_syn_tv = lam_tv * self._tv_loss(self.delta_od * region)

                # weak magnitude control so synthetic batches do not explode
                reg_den = torch.sum(region) + eps
                self.loss_G_syn_mag = lam_mag * (torch.sum(torch.abs(self.delta_od) * region) / reg_den)

                # optional weak anti-collapse term: encourage some non-zero response inside mask
                if lam_mask_mean > 0.0:
                    mean_inside = torch.sum(torch.abs(self.delta_od) * region) / reg_den
                    self.loss_G_syn_mask_mean = lam_mask_mean * torch.relu(0.05 - mean_inside)

        self.loss_G = (
            self.loss_G_GAN
            + self.loss_G_L1
            + self.loss_G_delta_bg
            + self.loss_G_delta
            + self.loss_G_grad
            + self.loss_G_delta_grad
            + self.loss_G_lap
            + self.loss_G_ssim
            + self.loss_G_stats
            + self.loss_G_syn_tv
            + self.loss_G_syn_mag
            + self.loss_G_syn_mask_mean
        )
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        # D: only for real paired samples
        if (not self.is_synthetic) and self.has_real_B:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
        else:
            z = torch.tensor(0.0, device=self.device)
            self.loss_D_fake = z.clone()
            self.loss_D_real = z.clone()
            self.loss_D = z.clone()

        # G: always train
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()