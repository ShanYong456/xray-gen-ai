import torch
import torch.nn.functional as F

from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    """pix2pix model with OPTIONAL physics-style delta compositing on an empty tray E.

    Input:
      A_cond + E
      where A_cond = [class one-hot channels, thickness channel, appearance(optional)]

    Target:
      B (real X-ray or pseudo target)

    Real paired sample:
      - GAN + supervised losses

    Synthetic pseudo-target sample:
      - NO GAN / NO D update
      - YES supervised losses if B is provided

    Mask-only synthetic sample:
      - NO GAN
      - NO paired supervision
      - only weak regularization
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
                            help="scale applied to predicted residual delta before adding in OD-space")

        parser.add_argument("--delta_positive", action="store_true",
                            help="force delta_od >= 0 using softplus (recommended for X-ray attenuation).")
        parser.add_argument("--delta_max", type=float, default=6.0,
                            help="Clamp OD delta to [0, delta_max] to prevent over-darkening/explosions.")

        parser.add_argument("--od_gamma", type=float, default=1.0,
                            help="Gamma applied before OD: OD = -log(I^gamma + eps). <1 softens OD.")

        # Conditioning layout
        # Conditioning layout (current)
        # ch0 = object mask
        # ch1 = appearance (optional)
        # ch2 = thickness (optional)
        parser.add_argument("--class_nc", type=int, default=1,
                            help="Number of mask/class channels in A. For current setup use 1.")
        parser.add_argument("--use_appearance_channel", action="store_true",
                            help="If set, A includes masked real-item appearance channel.")
        parser.add_argument("--appearance_nc", type=int, default=1,
                            help="Number of appearance channels in A. Current implementation supports 1.")
        parser.add_argument("--use_thickness_channel", action="store_true",
                            help="If set, A includes thickness proxy channel(s).")
        parser.add_argument("--thickness_nc", type=int, default=1,
                            help="Number of thickness channels in A.")
        parser.add_argument("--mask_thr", type=float, default=0.05,
                            help="threshold in [0,1] after unnormalizing class channels to detect object pixels")

        # Physics prior from thickness
        parser.add_argument("--use_delta_prior", action="store_true",
                            help="Use simple class-weighted thickness prior for delta_od.")
        parser.add_argument("--prior_shampoo", type=float, default=1.2,
                            help="Thickness prior strength for Shampoo.")
        parser.add_argument("--prior_blade", type=float, default=0.8,
                            help="Thickness prior strength for Blade.")
        parser.add_argument("--lambda_instance_delta", type=float, default=0.0,
                            help="Extra instance-wise delta supervision weight if instance_masks are provided.")

        # Masked L1 option
        parser.add_argument("--use_masked_l1", action="store_true",
                            help="use masked object L1 + background identity L1 (requires use_delta_comp).")
        parser.add_argument("--lambda_bg", type=float, default=1.5,
                            help="background identity strength vs empty tray outside mask")

        # Regularize delta outside mask to be ~0
        parser.add_argument("--lambda_delta_bg", type=float, default=5.0,
                            help="Penalty weight for delta outside object mask.")

        # Direct delta supervision
        parser.add_argument("--use_delta_supervision", action="store_true",
                            help="supervise delta_od with delta_gt = OD(B)-OD(E) inside mask.")
        parser.add_argument("--lambda_delta", type=float, default=50.0,
                            help="Weight for delta supervision loss.")

        # Tray constraint
        parser.add_argument("--use_tray_mask", action="store_true",
                            help="expect input['T'] tray mask (1 inside tray, 0 outside).")

        # Detail supervision
        parser.add_argument("--use_grad_loss", action="store_true",
                            help="Sobel gradient loss on fake_B vs real_B inside object region.")
        parser.add_argument("--lambda_grad", type=float, default=10.0,
                            help="Weight for gradient loss on fake_B vs real_B.")

        parser.add_argument("--use_delta_grad_loss", action="store_true",
                            help="Sobel gradient loss on delta_od vs delta_gt inside object region.")
        parser.add_argument("--lambda_delta_grad", type=float, default=10.0,
                            help="Weight for gradient loss on delta_od vs delta_gt.")

        parser.add_argument("--use_lap_loss", action="store_true",
                            help="Laplacian loss on fake_B vs real_B inside object region.")
        parser.add_argument("--lambda_lap", type=float, default=10.0,
                            help="Weight for Laplacian loss.")

        parser.add_argument("--use_ssim_loss", action="store_true",
                            help="SSIM loss on fake_B vs real_B inside object region.")
        parser.add_argument("--lambda_ssim", type=float, default=5.0,
                            help="Weight for SSIM loss.")

        parser.add_argument("--use_region_stats", action="store_true",
                            help="Match mean/std of fake_B to real_B inside object region.")
        parser.add_argument("--lambda_stats", type=float, default=5.0,
                            help="Weight for region stats loss.")

        # Soft mask options
        parser.add_argument("--use_soft_mask", action="store_true",
                            help="Use soft mask instead of binary mask.")
        parser.add_argument("--mask_soft_beta", type=float, default=30.0,
                            help="Soft mask sharpness.")

        parser.add_argument("--mask_blur_ksize", type=int, default=0,
                            help="Gaussian blur kernel size for mask smoothing (0 disables).")
        parser.add_argument("--mask_blur_sigma", type=float, default=1.2,
                            help="Gaussian blur sigma for mask smoothing.")
        parser.add_argument("--mask_noise_std", type=float, default=0.0,
                            help="Small noise added to conditioning channels during training.")

        # Synthetic-only regularization
        parser.add_argument("--lambda_syn_tv", type=float, default=1.0,
                            help="TV smoothness on delta_od for mask-only synthetic batches.")
        parser.add_argument("--lambda_syn_mag", type=float, default=0.2,
                            help="Weak magnitude regularization on delta_od for mask-only synthetic batches.")
        parser.add_argument("--lambda_syn_mask_mean", type=float, default=0.0,
                            help="Optional weak encouragement for non-zero response inside object mask.")

        parser.add_argument(
            "--pretrained_netG",
            type=str,
            default="",
            help="Path to pretrained old netG checkpoint to partially load."
        )

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
        self.visual_names = ["real_A", "fake_B"]

        self.model_names = ["G", "D"] if self.isTrain else ["G"]
        self.device = opt.device

        self.class_nc = int(getattr(opt, "class_nc", 1))
        self.thickness_nc = int(getattr(opt, "thickness_nc", 1))
        self.use_thickness_channel = bool(getattr(opt, "use_thickness_channel", False))
        self.use_appearance_channel = bool(getattr(opt, "use_appearance_channel", False))
        self.appearance_nc = int(getattr(opt, "appearance_nc", 1))

        if self.use_appearance_channel and self.appearance_nc != 1:
            raise ValueError("Current implementation supports --appearance_nc 1 only.")

        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
            not opt.no_dropout, opt.init_type, opt.init_gain
        )

        pretrained_netG = str(getattr(opt, "pretrained_netG", "")).strip()
        if pretrained_netG:
            self.load_pretrained_8ch_to_9ch(pretrained_netG)

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

        self.empty_E = None
        self.delta_raw = None
        self.delta_od = None
        self.delta_gt = None
        self.mask_M = None
        self.tray_T = None
        self.instance_masks = None

        self.is_synthetic = False
        self.has_real_B = False

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
        kx = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3) / 4.0
        ky = torch.tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3) / 4.0
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

        if self.isTrain:
            noise_std = float(getattr(self.opt, "mask_noise_std", 0.0))
            if noise_std > 0:
                # small noise to appearance only
                if self.use_appearance_channel:
                    app_start = self._appearance_channel_start_idx()
                    app_end = app_start + self.appearance_nc
                    if A.shape[1] >= app_end:
                        noise = torch.randn_like(A[:, app_start:app_end]) * noise_std
                        A[:, app_start:app_end] = torch.clamp(A[:, app_start:app_end] + noise, -1.0, 1.0)

        is_synth = input.get("is_synthetic", False)
        if isinstance(is_synth, torch.Tensor):
            is_synth = bool(is_synth.flatten()[0].item())
        self.is_synthetic = bool(is_synth)

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

        if getattr(self.opt, "use_tray_mask", False):
            if "T" not in input:
                raise KeyError("use_tray_mask set but input has no key 'T' (tray mask).")
            self.tray_T = input["T"].to(self.device)
        else:
            self.tray_T = None

        inst = input.get("instance_masks", None)
        if inst is not None:
            self.instance_masks = inst.to(self.device)
        else:
            self.instance_masks = None

        self.image_paths = input.get("A_paths" if AtoB else "B_paths", [])

    def _appearance_channel_start_idx(self):
        start = 1
        if bool(getattr(self.opt, "use_edge_channel", False)):
            start += 1
        if self.use_thickness_channel:
            start += self.thickness_nc
        if bool(getattr(self.opt, "use_coord_channels", False)):
            start += 2
        return start


    def _build_object_mask_from_A(self):
        """
        Build object mask.

        Priority:
        1. Use instance masks if available
        2. Otherwise fallback to explicit object-mask channel
        """
        if self.instance_masks is not None:
            inst = self.instance_masks

            if inst.dim() == 4:
                M = torch.clamp(inst.sum(dim=1, keepdim=True), 0.0, 1.0)
            elif inst.dim() == 3:
                M = inst.unsqueeze(1)
            else:
                raise RuntimeError(f"Unexpected instance_masks shape: {inst.shape}")

            blur_k = int(getattr(self.opt, "mask_blur_ksize", 0))
            blur_sigma = float(getattr(self.opt, "mask_blur_sigma", 1.2))

            if blur_k > 1:
                M = self._gaussian_blur(M, blur_k, blur_sigma)

            M = torch.clamp(M, 0.0, 1.0)
            return None, M.float()

        A_mask = self.real_A[:, :1, :, :]
        A01 = (A_mask + 1.0) * 0.5

        thr = float(getattr(self.opt, "mask_thr", 0.05))

        if getattr(self.opt, "use_soft_mask", False):
            beta = float(getattr(self.opt, "mask_soft_beta", 30.0))
            M = torch.sigmoid(beta * (A01 - thr))
        else:
            M = (A01 > thr).float()

        blur_k = int(getattr(self.opt, "mask_blur_ksize", 0))
        blur_sigma = float(getattr(self.opt, "mask_blur_sigma", 1.2))

        if blur_k > 1:
            M = self._gaussian_blur(M, blur_k, blur_sigma)

        M = torch.clamp(M, 0.0, 1.0)

        return A_mask, M

    def _get_thickness_from_A(self):
        if not self.use_thickness_channel:
            return None

        # New conditioning layout:
        # ch0 = mask
        # +1 edge if enabled
        # +1 thickness if enabled
        # +2 coords if enabled
        start = 1
        if bool(getattr(self.opt, "use_edge_channel", False)):
            start += 1

        end = start + self.thickness_nc

        if self.real_A.shape[1] < end:
            return None

        th = self.real_A[:, start:end, :, :]
        th = (th + 1.0) * 0.5
        if th.shape[1] > 1:
            th = th.mean(dim=1, keepdim=True)
        return torch.clamp(th, 0.0, 1.0)

    def _get_appearance_from_A(self):
        if not self.use_appearance_channel:
            return None

        start = 1  # mask
        if bool(getattr(self.opt, "use_edge_channel", False)):
            start += 1
        if self.use_thickness_channel:
            start += self.thickness_nc
        if bool(getattr(self.opt, "use_coord_channels", False)):
            start += 2

        end = start + self.appearance_nc

        if self.real_A.shape[1] < end:
            return None

        app = self.real_A[:, start:end, :, :]
        app = (app + 1.0) * 0.5
        if app.shape[1] > 1:
            app = app.mean(dim=1, keepdim=True)
        return torch.clamp(app, 0.0, 1.0)

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

    def _tv_loss(self, x: torch.Tensor) -> torch.Tensor:
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        return dx + dy

    def _build_delta_prior(self):
        """
        Simplified prior for current conditioning layout:
          ch0 = object mask
          ch1 = edge (optional)
          next = thickness (optional)
          next = coord x/y (optional)
          next = appearance (optional)

        For Shampoo-only training, use:
          prior = mask * thickness * prior_shampoo

        This gives a coarse physically reasonable interior attenuation base
        even when appearance is absent.
        """
        if not bool(getattr(self.opt, "use_delta_prior", False)):
            return None
        if not self.use_thickness_channel:
            return None

        thickness = self._get_thickness_from_A()
        if thickness is None:
            return None

        _, M = self._build_object_mask_from_A()
        if M is None:
            return None

        k_shampoo = float(getattr(self.opt, "prior_shampoo", 1.2))

        prior_1ch = M * thickness * k_shampoo
        prior_3ch = prior_1ch.repeat(1, 3, 1, 1)
        return prior_3ch

    def forward(self):
        if not getattr(self.opt, "use_delta_comp", False):
            self.fake_B = self.netG(self.real_A)
            self.delta_raw = None
            self.delta_od = None
            self.delta_gt = None
            self.mask_M = None
            return

        self.delta_raw = self.netG(self.real_A)

        _, M = self._build_object_mask_from_A()
        if getattr(self.opt, "use_tray_mask", False) and self.tray_T is not None:
            M = M * self.tray_T
        self.mask_M = M

        eps = float(getattr(self.opt, "compose_eps", 1e-6))
        delta_scale = float(getattr(self.opt, "delta_scale", 1.0))
        delta_max = float(getattr(self.opt, "delta_max", 6.0))
        delta_positive = bool(getattr(self.opt, "delta_positive", False))
        od_gamma = float(getattr(self.opt, "od_gamma", 1.0))

        # Empty tray -> intensity [0,1] -> OD
        E01 = torch.clamp((self.empty_E + 1.0) * 0.5, 0.0, 1.0)
        if od_gamma != 1.0:
            E01 = torch.pow(E01, od_gamma)
        OD_E = -torch.log(E01 + eps)

        # Generator residual
        if delta_positive:
            delta_res = F.softplus(self.delta_raw)
        else:
            delta_res = self.delta_raw

        delta_res = delta_res * delta_scale

        # Optional physics prior from thickness/classes
        delta_prior = self._build_delta_prior()
        if delta_prior is not None:
            delta_od = delta_prior + delta_res
        else:
            delta_od = delta_res

        # Safer delta range control
        if delta_positive:
            delta_od = torch.clamp(delta_od, 0.0, delta_max)
        else:
            # allow some brightening if needed, but prevent runaway values
            delta_od = torch.clamp(delta_od, -delta_max, delta_max)

        # Restrict delta to tray if tray mask is used
        if getattr(self.opt, "use_tray_mask", False) and self.tray_T is not None:
            delta_od = delta_od * self._expand_like(self.tray_T, delta_od)

        self.delta_od = delta_od

        # Expand masks to match channels
        T = self.tray_T if (getattr(self.opt, "use_tray_mask", False) and self.tray_T is not None) else torch.ones_like(M)
        M_exp = self._expand_like(M, delta_od)
        T_exp = self._expand_like(T, delta_od)

        # Compose only inside object mask and tray
        compose_mask = T_exp * M_exp
        OD_pred = OD_E + compose_mask * delta_od

        # Back to intensity
        I_pred = torch.exp(-OD_pred)
        I_pred = torch.clamp(I_pred, 0.0, 1.0)

        # Outside tray, keep exact empty tray
        if getattr(self.opt, "use_tray_mask", False) and self.tray_T is not None:
            I_pred = T_exp * I_pred + (1.0 - T_exp) * E01

        self.fake_B = I_pred * 2.0 - 1.0

        # Optional GT delta supervision
        if getattr(self.opt, "use_delta_supervision", False) and self.has_real_B:
            B01 = torch.clamp((self.real_B + 1.0) * 0.5, 0.0, 1.0)
            if od_gamma != 1.0:
                B01 = torch.pow(B01, od_gamma)
            OD_B = -torch.log(B01 + eps)

            delta_gt = OD_B - OD_E

            if delta_positive:
                delta_gt = torch.clamp(delta_gt, 0.0, delta_max)
            else:
                delta_gt = torch.clamp(delta_gt, -delta_max, delta_max)

            if getattr(self.opt, "use_tray_mask", False) and self.tray_T is not None:
                delta_gt = delta_gt * self._expand_like(self.tray_T, delta_gt)

            self.delta_gt = delta_gt
        else:
            self.delta_gt = None

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

        _, M_obj_1 = self._build_object_mask_from_A()
        if getattr(self.opt, "use_tray_mask", False) and self.tray_T is not None:
            M_obj_1 = M_obj_1 * self.tray_T

        T_1 = self.tray_T if (getattr(self.opt, "use_tray_mask", False) and self.tray_T is not None) else torch.ones_like(M_obj_1)
        M_obj = self._expand_like(M_obj_1, self.fake_B)
        T = self._expand_like(T_1, self.fake_B)

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

                lam_inst = float(getattr(self.opt, "lambda_instance_delta", 0.0))
                if lam_inst > 0.0 and self.instance_masks is not None and self.instance_masks.numel() > 0:
                    inst_losses = []
                    if self.instance_masks.dim() == 4:
                        Bn, N, H, W = self.instance_masks.shape
                        for i in range(N):
                            Mi = self.instance_masks[:, i:i+1, :, :]
                            if getattr(self.opt, "use_tray_mask", False) and self.tray_T is not None:
                                Mi = Mi * self.tray_T
                            region_i = self._expand_like(Mi, self.delta_od)
                            den_i = torch.sum(region_i) + eps
                            if den_i.item() <= 1e-6:
                                continue
                            li = torch.sum(torch.abs(self.delta_od - self.delta_gt) * region_i) / den_i
                            inst_losses.append(li)
                    elif self.instance_masks.dim() == 3:
                        Mi = self.instance_masks.unsqueeze(1)
                        region_i = self._expand_like(Mi, self.delta_od)
                        den_i = torch.sum(region_i) + eps
                        if den_i.item() > 1e-6:
                            inst_losses.append(torch.sum(torch.abs(self.delta_od - self.delta_gt) * region_i) / den_i)

                    if len(inst_losses) > 0:
                        self.loss_G_delta = self.loss_G_delta + lam_inst * torch.stack(inst_losses).mean()

        if self.has_real_B and getattr(self.opt, "use_grad_loss", False):
            lam_grad = float(getattr(self.opt, "lambda_grad", 10.0))
            eps = 1e-6
            g_fake = self._sobel_mag(self.fake_B)
            g_real = self._sobel_mag(self.real_B)
            region = M_obj * T
            num = torch.sum(torch.abs(g_fake - g_real) * region)
            den = torch.sum(region) + eps
            self.loss_G_grad = lam_grad * (num / den)

        if self.delta_gt is not None and self.delta_od is not None and getattr(self.opt, "use_delta_grad_loss", False):
            lam_dg = float(getattr(self.opt, "lambda_delta_grad", 10.0))
            eps = 1e-6
            g_d = self._sobel_mag(self.delta_od)
            g_gt = self._sobel_mag(self.delta_gt)
            region = self._expand_like(M_obj_1, self.delta_od) * self._expand_like(T_1, self.delta_od)
            num = torch.sum(torch.abs(g_d - g_gt) * region)
            den = torch.sum(region) + eps
            self.loss_G_delta_grad = lam_dg * (num / den)

        if self.has_real_B and getattr(self.opt, "use_lap_loss", False):
            lam_lap = float(getattr(self.opt, "lambda_lap", 10.0))
            eps = 1e-6
            lap_fake = self._laplacian(self.fake_B)
            lap_real = self._laplacian(self.real_B)
            region = M_obj * T
            num = torch.sum(torch.abs(lap_fake - lap_real) * region)
            den = torch.sum(region) + eps
            self.loss_G_lap = lam_lap * (num / den)

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

        if (not self.has_real_B) and self.delta_od is not None:
            lam_tv = float(getattr(self.opt, "lambda_syn_tv", 1.0))
            lam_mag = float(getattr(self.opt, "lambda_syn_mag", 0.2))
            lam_mask_mean = float(getattr(self.opt, "lambda_syn_mask_mean", 0.0))

            Mch = self._expand_like(M_obj_1, self.delta_od)
            Tch = self._expand_like(T_1, self.delta_od)
            region = Mch * Tch
            eps = 1e-6

            self.loss_G_syn_tv = lam_tv * self._tv_loss(self.delta_od * region)

            reg_den = torch.sum(region) + eps
            self.loss_G_syn_mag = lam_mag * (torch.sum(torch.abs(self.delta_od) * region) / reg_den)

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

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
    

    def load_pretrained_8ch_to_9ch(self, pretrained_path: str):
        """
        Load old 8-channel checkpoint into current 9-channel netG.
        Copies all matching weights.
        For first conv layer:
        old shape = [out_c, 8, k, k]
        new shape = [out_c, 9, k, k]
        We copy first 8 channels and zero-init the new appearance channel.
        """
        state = torch.load(pretrained_path, map_location=self.device)

        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        net_state = self.netG.state_dict()
        new_state = {}

        for k, v in state.items():
            if k not in net_state:
                continue

            if net_state[k].shape == v.shape:
                new_state[k] = v
                continue

            # Special handling for first conv weight
            if "model.model.0.weight" in k or k == "model.model.0.weight":
                old_w = v
                new_w = net_state[k].clone()

                if old_w.ndim == 4 and new_w.ndim == 4:
                    out_c_new, in_c_new, kh_new, kw_new = new_w.shape
                    out_c_old, in_c_old, kh_old, kw_old = old_w.shape

                    if out_c_new == out_c_old and kh_new == kh_old and kw_new == kw_old and in_c_old == 8 and in_c_new == 9:
                        new_w[:, :8, :, :] = old_w
                        new_w[:, 8:9, :, :] = 0.0
                        new_state[k] = new_w
                        print(f"[pretrain] adapted first conv from {old_w.shape} -> {new_w.shape}")
                        continue

            print(f"[pretrain] skipped shape mismatch: {k} {v.shape} -> {net_state[k].shape}")

        missing, unexpected = self.netG.load_state_dict(new_state, strict=False)
        print(f"[pretrain] loaded netG from {pretrained_path}")
        print(f"[pretrain] missing keys: {len(missing)}")
        print(f"[pretrain] unexpected keys: {len(unexpected)}")

    def get_current_visuals(self):
        visuals = {}

        if hasattr(self, "real_A") and self.real_A is not None:
            visuals["real_A"] = self.real_A

        if hasattr(self, "fake_B") and self.fake_B is not None:
            visuals["fake_B"] = self.fake_B

        if hasattr(self, "real_B") and self.real_B is not None:
            visuals["real_B"] = self.real_B

        if hasattr(self, "empty_E") and self.empty_E is not None:
            visuals["E"] = self.empty_E

        if hasattr(self, "tray_T") and self.tray_T is not None:
            visuals["T"] = self.tray_T

        return visuals