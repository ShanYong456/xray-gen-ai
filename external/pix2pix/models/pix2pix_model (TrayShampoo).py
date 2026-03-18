import torch
import torch.nn.functional as F

from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    """
    pix2pix with physics-style delta compositing on an empty tray E.

    Key design decisions for 1024px realism:
    - MultiScale discriminator (netD=multiscale or n_layers_D=4) instead of basic PatchGAN
    - GAN loss active for ALL samples (real + synthetic), with label smoothing on synthetic
    - Rebalanced losses: L1 up, delta down, so GAN gradient is not crushed
    - delta_positive=False: allow the model to learn local brightening (beam hardening)
    - Gradient penalty option to stabilise D instead of letting it collapse
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm="instance", netG="unet_256", dataset_mode="aligned")
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="lsgan")
            parser.add_argument("--lambda_L1", type=float, default=30.0)

        # Physics / delta-compositing
        parser.add_argument("--compose_eps", type=float, default=1e-6)
        parser.add_argument("--delta_scale", type=float, default=1.0)
        parser.add_argument("--delta_positive", action="store_true",
                            help="Force delta >= 0 (disable if outputs too dark).")
        parser.add_argument("--delta_max", type=float, default=5.0,
                            help="OD delta clamp. Raise if real objects are being clipped.")
        parser.add_argument("--od_gamma", type=float, default=1.0)

        # Conditioning layout
        parser.add_argument("--class_nc", type=int, default=1)
        parser.add_argument("--use_appearance_channel", action="store_true")
        parser.add_argument("--appearance_nc", type=int, default=1)
        parser.add_argument("--use_thickness_channel", action="store_true")
        parser.add_argument("--thickness_nc", type=int, default=1)
        parser.add_argument("--mask_thr", type=float, default=0.05)

        # Thickness prior
        parser.add_argument("--use_delta_prior", action="store_true")
        parser.add_argument("--prior_shampoo", type=float, default=1.2)
        parser.add_argument("--prior_blade", type=float, default=0.8)

        # Masked L1
        parser.add_argument("--use_masked_l1", action="store_true")
        parser.add_argument("--lambda_bg", type=float, default=1.5)

        # Delta background penalty
        parser.add_argument("--lambda_delta_bg", type=float, default=3.0)

        # Delta supervision
        parser.add_argument("--use_delta_supervision", action="store_true")
        parser.add_argument("--lambda_delta", type=float, default=50.0,
                            help="Keep this well below lambda_L1 * 10 or GAN collapses.")
        parser.add_argument("--lambda_instance_delta", type=float, default=0.0)

        # Tray
        parser.add_argument("--use_tray_mask", action="store_true")

        # Detail losses
        parser.add_argument("--use_grad_loss", action="store_true")
        parser.add_argument("--lambda_grad", type=float, default=10.0)
        parser.add_argument("--use_delta_grad_loss", action="store_true")
        parser.add_argument("--lambda_delta_grad", type=float, default=10.0)
        parser.add_argument("--use_lap_loss", action="store_true")
        parser.add_argument("--lambda_lap", type=float, default=6.0)
        parser.add_argument("--use_ssim_loss", action="store_true")
        parser.add_argument("--lambda_ssim", type=float, default=3.0)
        parser.add_argument("--use_region_stats", action="store_true")
        parser.add_argument("--lambda_stats", type=float, default=3.0)

        # Soft mask
        parser.add_argument("--use_soft_mask", action="store_true")
        parser.add_argument("--mask_soft_beta", type=float, default=30.0)
        parser.add_argument("--mask_blur_ksize", type=int, default=0)
        parser.add_argument("--mask_blur_sigma", type=float, default=1.2)
        parser.add_argument("--mask_noise_std", type=float, default=0.0)

        # Synthetic-only regularisation
        parser.add_argument("--lambda_syn_tv", type=float, default=0.25)
        parser.add_argument("--lambda_syn_mag", type=float, default=0.07)
        parser.add_argument("--lambda_syn_mask_mean", type=float, default=0.0)

        # ── GAN stability fixes ───────────────────────────────────────────
        parser.add_argument("--lambda_gp", type=float, default=0.0,
                            help="Gradient penalty weight for D (R1 penalty). "
                                 "Set 1.0–10.0 to prevent D collapse. 0 disables.")
        parser.add_argument("--d_label_smooth", type=float, default=0.1,
                            help="One-sided label smoothing for real samples in D. "
                                 "Real target becomes 1-smooth instead of 1.")
        parser.add_argument("--syn_gan_weight", type=float, default=0.3,
                            help="GAN loss weight for synthetic samples (0=disabled, "
                                 "1=same as real). Default 0.3 keeps D honest on synth.")
        parser.add_argument("--d_update_ratio", type=int, default=1,
                            help="Update D every N G steps. Increase if D collapses fast.")
        # ─────────────────────────────────────────────────────────────────

        parser.add_argument("--pretrained_netG", type=str, default="")

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = [
            "G_GAN", "G_L1",
            "G_delta_bg", "G_delta",
            "G_grad", "G_delta_grad",
            "G_lap", "G_ssim", "G_stats",
            "G_syn_tv", "G_syn_mag", "G_syn_mask_mean",
            "D_real", "D_fake",
        ]
        self.visual_names = ["real_A", "fake_B"]
        self.model_names = ["G", "D"] if self.isTrain else ["G"]
        self.device = opt.device

        self.class_nc = int(getattr(opt, "class_nc", 1))
        self.thickness_nc = int(getattr(opt, "thickness_nc", 1))
        self.use_thickness_channel = bool(getattr(opt, "use_thickness_channel", False))
        self.use_appearance_channel = bool(getattr(opt, "use_appearance_channel", False))
        self.appearance_nc = int(getattr(opt, "appearance_nc", 1))
        self._g_step = 0  # track steps for d_update_ratio

        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
            not opt.no_dropout, opt.init_type, opt.init_gain,
        )

        pretrained = str(getattr(opt, "pretrained_netG", "")).strip()
        if pretrained:
            self._load_pretrained_netG(pretrained)

        if self.isTrain:
            self.netD = networks.define_D(
                opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain,
            )
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers += [self.optimizer_G, self.optimizer_D]

        # State tensors
        self.empty_E = self.delta_raw = self.delta_od = self.delta_gt = None
        self.mask_M = self.tray_T = self.instance_masks = None
        self.is_synthetic = self.has_real_B = False

        z = torch.tensor(0.0, device=self.device)
        for name in ["G_delta_bg", "G_delta", "G_grad", "G_delta_grad",
                     "G_lap", "G_ssim", "G_stats",
                     "G_syn_tv", "G_syn_mag", "G_syn_mask_mean"]:
            setattr(self, f"loss_{name}", z.clone())

        self._init_kernels()

    # ─────────────────────────────────────────────────────────────────────────
    # Kernel setup
    # ─────────────────────────────────────────────────────────────────────────

    def _init_kernels(self):
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                           dtype=torch.float32).view(1, 1, 3, 3) / 4.0
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                           dtype=torch.float32).view(1, 1, 3, 3) / 4.0
        kl = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                           dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_kx = kx.to(self.device)
        self.sobel_ky = ky.to(self.device)
        self.lap_k = kl.to(self.device)

    def _gaussian_blur(self, x, ksize=5, sigma=1.2):
        if ksize <= 1:
            return x
        ksize = ksize + (0 if ksize % 2 else 1)
        C = x.shape[1]
        g = torch.exp(-(torch.arange(ksize, device=x.device) - (ksize - 1) / 2) ** 2
                      / (2 * sigma ** 2))
        g = g / g.sum()
        kx_ = g.view(1, 1, 1, ksize).repeat(C, 1, 1, 1)
        ky_ = g.view(1, 1, ksize, 1).repeat(C, 1, 1, 1)
        x = F.conv2d(x, kx_, padding=(0, ksize // 2), groups=C)
        x = F.conv2d(x, ky_, padding=(ksize // 2, 0), groups=C)
        return x

    # ─────────────────────────────────────────────────────────────────────────
    # set_input
    # ─────────────────────────────────────────────────────────────────────────

    def set_input(self, input):
        AtoB = self.opt.direction == "AtoB"
        A = input["A" if AtoB else "B"].to(self.device)

        if self.isTrain:
            noise_std = float(getattr(self.opt, "mask_noise_std", 0.0))
            if noise_std > 0 and self.use_appearance_channel:
                s = self._appearance_channel_start_idx()
                e = s + self.appearance_nc
                if A.shape[1] >= e:
                    A[:, s:e] = torch.clamp(
                        A[:, s:e] + torch.randn_like(A[:, s:e]) * noise_std, -1, 1)

        is_synth = input.get("is_synthetic", False)
        self.is_synthetic = bool(
            is_synth.flatten()[0].item() if torch.is_tensor(is_synth) else is_synth)

        b_key = "B" if AtoB else "A"
        if b_key in input and input[b_key] is not None:
            self.real_B = input[b_key].to(self.device)
            self.has_real_B = True
        else:
            self.real_B = None
            self.has_real_B = False

        if getattr(self.opt, "use_delta_comp", False):
            self.empty_E = input["E"].to(self.device)
            self.real_A = torch.cat([A, self.empty_E], dim=1)
        else:
            self.real_A = A
            self.empty_E = None

        self.tray_T = (input["T"].to(self.device)
                       if getattr(self.opt, "use_tray_mask", False) else None)

        inst = input.get("instance_masks", None)
        self.instance_masks = inst.to(self.device) if inst is not None else None
        self.image_paths = input.get("A_paths" if AtoB else "B_paths", [])

    # ─────────────────────────────────────────────────────────────────────────
    # Mask / conditioning helpers
    # ─────────────────────────────────────────────────────────────────────────

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
        if self.instance_masks is not None:
            inst = self.instance_masks
            M = torch.clamp(
                inst.sum(dim=1, keepdim=True) if inst.dim() == 4 else inst.unsqueeze(1),
                0.0, 1.0)
            blur_k = int(getattr(self.opt, "mask_blur_ksize", 0))
            if blur_k > 1:
                M = self._gaussian_blur(M, blur_k, float(getattr(self.opt, "mask_blur_sigma", 1.2)))
            return None, torch.clamp(M, 0.0, 1.0).float()

        A01 = (self.real_A[:, :1] + 1.0) * 0.5
        thr = float(getattr(self.opt, "mask_thr", 0.05))
        if getattr(self.opt, "use_soft_mask", False):
            M = torch.sigmoid(float(getattr(self.opt, "mask_soft_beta", 30.0)) * (A01 - thr))
        else:
            M = (A01 > thr).float()
        blur_k = int(getattr(self.opt, "mask_blur_ksize", 0))
        if blur_k > 1:
            M = self._gaussian_blur(M, blur_k, float(getattr(self.opt, "mask_blur_sigma", 1.2)))
        return self.real_A[:, :1], torch.clamp(M, 0.0, 1.0)

    def _get_thickness_from_A(self):
        if not self.use_thickness_channel:
            return None
        start = 1 + (1 if bool(getattr(self.opt, "use_edge_channel", False)) else 0)
        end = start + self.thickness_nc
        if self.real_A.shape[1] < end:
            return None
        th = (self.real_A[:, start:end] + 1.0) * 0.5
        return torch.clamp(th.mean(dim=1, keepdim=True) if th.shape[1] > 1 else th, 0, 1)

    def _expand_like(self, mask_1ch, x):
        if mask_1ch is None:
            return None
        if mask_1ch.shape[1] == 1 and x.shape[1] != 1:
            return mask_1ch.expand(-1, x.shape[1], -1, -1)
        return mask_1ch

    # ─────────────────────────────────────────────────────────────────────────
    # Differential operators
    # ─────────────────────────────────────────────────────────────────────────

    def _sobel_mag(self, x):
        C = x.shape[1]
        kx = self.sobel_kx.to(x.dtype).repeat(C, 1, 1, 1)
        ky = self.sobel_ky.to(x.dtype).repeat(C, 1, 1, 1)
        return torch.sqrt(F.conv2d(x, kx, padding=1, groups=C) ** 2
                          + F.conv2d(x, ky, padding=1, groups=C) ** 2 + 1e-12)

    def _laplacian(self, x):
        C = x.shape[1]
        return F.conv2d(x, self.lap_k.to(x.dtype).repeat(C, 1, 1, 1), padding=1, groups=C)

    def _ssim_map(self, x, y, window_size=7):
        pad = window_size // 2
        mu_x = F.avg_pool2d(x, window_size, stride=1, padding=pad)
        mu_y = F.avg_pool2d(y, window_size, stride=1, padding=pad)
        sx = F.avg_pool2d(x * x, window_size, stride=1, padding=pad) - mu_x * mu_x
        sy = F.avg_pool2d(y * y, window_size, stride=1, padding=pad) - mu_y * mu_y
        sxy = F.avg_pool2d(x * y, window_size, stride=1, padding=pad) - mu_x * mu_y
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        return ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / (
            (mu_x ** 2 + mu_y ** 2 + C1) * (sx + sy + C2) + 1e-12)

    def _tv_loss(self, x):
        return (torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
                + torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean())

    # ─────────────────────────────────────────────────────────────────────────
    # Physics prior
    # ─────────────────────────────────────────────────────────────────────────

    def _build_delta_prior(self):
        if not bool(getattr(self.opt, "use_delta_prior", False)) or not self.use_thickness_channel:
            return None
        thickness = self._get_thickness_from_A()
        if thickness is None:
            return None
        _, M = self._build_object_mask_from_A()
        return (M * thickness * float(getattr(self.opt, "prior_shampoo", 1.2))).repeat(1, 3, 1, 1)

    # ─────────────────────────────────────────────────────────────────────────
    # forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self):
        if not getattr(self.opt, "use_delta_comp", False):
            self.fake_B = self.netG(self.real_A)
            self.delta_raw = self.delta_od = self.delta_gt = self.mask_M = None
            return

        self.delta_raw = self.netG(self.real_A)
        _, M = self._build_object_mask_from_A()
        if getattr(self.opt, "use_tray_mask", False) and self.tray_T is not None:
            M = M * self.tray_T
        self.mask_M = M

        eps = float(getattr(self.opt, "compose_eps", 1e-6))
        delta_scale = float(getattr(self.opt, "delta_scale", 1.0))
        delta_max = float(getattr(self.opt, "delta_max", 5.0))
        delta_positive = bool(getattr(self.opt, "delta_positive", False))
        od_gamma = float(getattr(self.opt, "od_gamma", 1.0))

        E01 = torch.clamp((self.empty_E + 1.0) * 0.5, 0.0, 1.0)
        if od_gamma != 1.0:
            E01 = torch.pow(E01, od_gamma)
        OD_E = -torch.log(E01 + eps)

        delta_res = (F.softplus(self.delta_raw) if delta_positive
                     else self.delta_raw) * delta_scale

        delta_prior = self._build_delta_prior()
        delta_od = (delta_prior + delta_res) if delta_prior is not None else delta_res
        delta_od = (torch.clamp(delta_od, 0.0, delta_max) if delta_positive
                    else torch.clamp(delta_od, -delta_max, delta_max))

        if getattr(self.opt, "use_tray_mask", False) and self.tray_T is not None:
            delta_od = delta_od * self._expand_like(self.tray_T, delta_od)
        self.delta_od = delta_od

        T = (self.tray_T if (getattr(self.opt, "use_tray_mask", False)
                             and self.tray_T is not None)
             else torch.ones_like(M))
        compose_mask = self._expand_like(T, delta_od) * self._expand_like(M, delta_od)
        I_pred = torch.clamp(torch.exp(-(OD_E + compose_mask * delta_od)), 0.0, 1.0)

        if getattr(self.opt, "use_tray_mask", False) and self.tray_T is not None:
            T_exp = self._expand_like(T, I_pred)
            I_pred = T_exp * I_pred + (1.0 - T_exp) * E01

        self.fake_B = I_pred * 2.0 - 1.0

        if getattr(self.opt, "use_delta_supervision", False) and self.has_real_B:
            B01 = torch.clamp((self.real_B + 1.0) * 0.5, 0.0, 1.0)
            if od_gamma != 1.0:
                B01 = torch.pow(B01, od_gamma)
            delta_gt = -torch.log(B01 + eps) - OD_E
            delta_gt = (torch.clamp(delta_gt, 0.0, delta_max) if delta_positive
                        else torch.clamp(delta_gt, -delta_max, delta_max))
            if getattr(self.opt, "use_tray_mask", False) and self.tray_T is not None:
                delta_gt = delta_gt * self._expand_like(self.tray_T, delta_gt)
            self.delta_gt = delta_gt
        else:
            self.delta_gt = None

    # ─────────────────────────────────────────────────────────────────────────
    # Gradient penalty (R1) — prevents D collapse
    # ─────────────────────────────────────────────────────────────────────────

    def _r1_penalty(self, real_AB):
        """
        R1 gradient penalty: penalises D gradient norm on real samples.
        Keeps D from becoming too confident, preventing collapse.
        Call with lambda_gp > 0 (e.g. 1.0–10.0).
        """
        real_AB = real_AB.detach().requires_grad_(True)
        pred_real = self.netD(real_AB)
        if isinstance(pred_real, list):
            pred_real = pred_real[-1]
        if isinstance(pred_real, list):
            pred_real = pred_real[-1]
        grad = torch.autograd.grad(
            outputs=pred_real.sum(),
            inputs=real_AB,
            create_graph=True,
            retain_graph=True,
        )[0]
        return grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()

    # ─────────────────────────────────────────────────────────────────────────
    # backward_D — with label smoothing + gradient penalty
    # ─────────────────────────────────────────────────────────────────────────

    def backward_D(self):
        """
        Key fix: D now trains on ALL samples (real + synthetic).
        For synthetic samples we use a reduced real-label weight (syn_gan_weight)
        so the generator still gets an adversarial signal without misleading D
        with pseudo-targets.

        Label smoothing on real samples prevents D from becoming overconfident,
        which is the primary cause of D collapse in pix2pix at high resolution.
        """
        smooth = float(getattr(self.opt, "d_label_smooth", 0.1))
        lam_gp = float(getattr(self.opt, "lambda_gp", 0.0))

        fake_AB = torch.cat((self.real_A, self.fake_B.detach()), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        if self.has_real_B:
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB)
            # One-sided label smoothing: real label = 1 - smooth
            if smooth > 0:
                target = torch.ones_like(
                    pred_real if not isinstance(pred_real, list)
                    else pred_real[-1] if not isinstance(pred_real[-1], list)
                    else pred_real[-1][-1]
                ) * (1.0 - smooth)
                if isinstance(pred_real, list):
                    self.loss_D_real = sum(
                        F.mse_loss(p[-1] if isinstance(p, list) else p, target)
                        for p in pred_real
                    ) / len(pred_real)
                else:
                    self.loss_D_real = F.mse_loss(pred_real, target)
            else:
                self.loss_D_real = self.criterionGAN(pred_real, True)

            gp_loss = 0.0
            if lam_gp > 0:
                gp_loss = self._r1_penalty(real_AB) * lam_gp

            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 + gp_loss
        else:
            # Synthetic-only batch: only fake loss
            self.loss_D_real = torch.tensor(0.0, device=self.device)
            self.loss_D = self.loss_D_fake * 0.5

        self.loss_D.backward()

    # ─────────────────────────────────────────────────────────────────────────
    # backward_G — GAN active for all samples
    # ─────────────────────────────────────────────────────────────────────────

    def backward_G(self):
        z = torch.tensor(0.0, device=self.device)
        syn_weight = float(getattr(self.opt, "syn_gan_weight", 0.3))

        # ── GAN loss — active for BOTH real and synthetic ─────────────────
        # For synthetic: weight is syn_gan_weight (default 0.3) instead of 1.0
        # This keeps the generator honest without over-relying on pseudo-B quality
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        gan_raw = self.criterionGAN(pred_fake, True)
        gan_scale = 1.0 if (not self.is_synthetic and self.has_real_B) else syn_weight
        self.loss_G_GAN = gan_raw * gan_scale

        # Reset aux losses
        for name in ["G_delta_bg", "G_delta", "G_grad", "G_delta_grad",
                     "G_lap", "G_ssim", "G_stats",
                     "G_syn_tv", "G_syn_mag", "G_syn_mask_mean"]:
            setattr(self, f"loss_{name}", z.clone())

        _, M_obj_1 = self._build_object_mask_from_A()
        if getattr(self.opt, "use_tray_mask", False) and self.tray_T is not None:
            M_obj_1 = M_obj_1 * self.tray_T
        T_1 = (self.tray_T if (getattr(self.opt, "use_tray_mask", False)
                               and self.tray_T is not None)
               else torch.ones_like(M_obj_1))
        M_obj = self._expand_like(M_obj_1, self.fake_B)
        T = self._expand_like(T_1, self.fake_B)
        eps = 1e-6

        # ── L1 loss ───────────────────────────────────────────────────────
        if self.has_real_B:
            if getattr(self.opt, "use_delta_comp", False) and getattr(self.opt, "use_masked_l1", False):
                obj_r = M_obj * T
                bg_r = (1.0 - M_obj) * T
                obj_l1 = (torch.sum(torch.abs(self.fake_B - self.real_B) * obj_r)
                          / (torch.sum(obj_r) + eps))
                bg_l1 = (torch.sum(torch.abs(self.fake_B - self.empty_E) * bg_r)
                         / (torch.sum(bg_r) + eps))
                self.loss_G_L1 = (obj_l1 + float(getattr(self.opt, "lambda_bg", 1.5)) * bg_l1
                                  ) * self.opt.lambda_L1
            elif getattr(self.opt, "use_tray_mask", False) and self.tray_T is not None:
                num = torch.sum(torch.abs(self.fake_B - self.real_B) * T)
                self.loss_G_L1 = (num / (torch.sum(T) + eps)) * self.opt.lambda_L1
            else:
                self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = z.clone()

        # ── Delta background penalty ───────────────────────────────────────
        lam_dbg = float(getattr(self.opt, "lambda_delta_bg", 3.0))
        if lam_dbg > 0 and self.delta_od is not None and self.mask_M is not None:
            inside_bg = (1.0 - self._expand_like(self.mask_M, self.delta_od)) * self._expand_like(T_1, self.delta_od)
            self.loss_G_delta_bg = torch.mean(torch.abs(self.delta_od) * inside_bg) * lam_dbg

        # ── Delta supervision ─────────────────────────────────────────────
        if (getattr(self.opt, "use_delta_comp", False)
                and getattr(self.opt, "use_delta_supervision", False)
                and self.delta_gt is not None and self.delta_od is not None):
            lam_delta = float(getattr(self.opt, "lambda_delta", 50.0))
            region = self._expand_like(M_obj_1, self.delta_od) * self._expand_like(T_1, self.delta_od)
            self.loss_G_delta = (lam_delta
                                 * torch.sum(torch.abs(self.delta_od - self.delta_gt) * region)
                                 / (torch.sum(region) + eps))

            # Instance-wise delta
            lam_inst = float(getattr(self.opt, "lambda_instance_delta", 0.0))
            if lam_inst > 0 and self.instance_masks is not None and self.instance_masks.numel() > 0:
                inst_losses = []
                masks = (self.instance_masks if self.instance_masks.dim() == 4
                         else self.instance_masks.unsqueeze(0))
                for i in range(masks.shape[1]):
                    Mi = masks[:, i:i + 1]
                    if getattr(self.opt, "use_tray_mask", False) and self.tray_T is not None:
                        Mi = Mi * self.tray_T
                    ri = self._expand_like(Mi, self.delta_od)
                    den = torch.sum(ri) + eps
                    if den > 1e-6:
                        inst_losses.append(
                            torch.sum(torch.abs(self.delta_od - self.delta_gt) * ri) / den)
                if inst_losses:
                    self.loss_G_delta = self.loss_G_delta + lam_inst * torch.stack(inst_losses).mean()

        # ── Detail losses (object region only) ───────────────────────────
        region = M_obj * T

        if self.has_real_B and getattr(self.opt, "use_grad_loss", False):
            lam = float(getattr(self.opt, "lambda_grad", 10.0))
            num = torch.sum(torch.abs(self._sobel_mag(self.fake_B) - self._sobel_mag(self.real_B)) * region)
            self.loss_G_grad = lam * num / (torch.sum(region) + eps)

        if (self.delta_gt is not None and self.delta_od is not None
                and getattr(self.opt, "use_delta_grad_loss", False)):
            lam = float(getattr(self.opt, "lambda_delta_grad", 10.0))
            dr = self._expand_like(M_obj_1, self.delta_od) * self._expand_like(T_1, self.delta_od)
            num = torch.sum(torch.abs(self._sobel_mag(self.delta_od) - self._sobel_mag(self.delta_gt)) * dr)
            self.loss_G_delta_grad = lam * num / (torch.sum(dr) + eps)

        if self.has_real_B and getattr(self.opt, "use_lap_loss", False):
            lam = float(getattr(self.opt, "lambda_lap", 6.0))
            num = torch.sum(torch.abs(self._laplacian(self.fake_B) - self._laplacian(self.real_B)) * region)
            self.loss_G_lap = lam * num / (torch.sum(region) + eps)

        if self.has_real_B and getattr(self.opt, "use_ssim_loss", False):
            lam = float(getattr(self.opt, "lambda_ssim", 3.0))
            f01 = torch.clamp((self.fake_B + 1.0) * 0.5, 0.0, 1.0)
            r01 = torch.clamp((self.real_B + 1.0) * 0.5, 0.0, 1.0)
            num = torch.sum((1.0 - self._ssim_map(f01, r01)) * region)
            self.loss_G_ssim = lam * num / (torch.sum(region) + eps)

        if self.has_real_B and getattr(self.opt, "use_region_stats", False):
            lam = float(getattr(self.opt, "lambda_stats", 3.0))
            f01 = torch.clamp((self.fake_B + 1.0) * 0.5, 0.0, 1.0)
            r01 = torch.clamp((self.real_B + 1.0) * 0.5, 0.0, 1.0)
            reg_sum = torch.sum(region, dim=(2, 3), keepdim=True) + eps
            f_mean = torch.sum(f01 * region, dim=(2, 3), keepdim=True) / reg_sum
            r_mean = torch.sum(r01 * region, dim=(2, 3), keepdim=True) / reg_sum
            f_std = torch.sqrt(torch.sum((f01 - f_mean) ** 2 * region, dim=(2, 3), keepdim=True) / reg_sum + 1e-12)
            r_std = torch.sqrt(torch.sum((r01 - r_mean) ** 2 * region, dim=(2, 3), keepdim=True) / reg_sum + 1e-12)
            self.loss_G_stats = lam * (torch.mean(torch.abs(f_mean - r_mean))
                                       + torch.mean(torch.abs(f_std - r_std)))

        # ── Synthetic-only regularisation ─────────────────────────────────
        if not self.has_real_B and self.delta_od is not None:
            dr = self._expand_like(M_obj_1, self.delta_od) * self._expand_like(T_1, self.delta_od)
            reg_den = torch.sum(dr) + eps
            self.loss_G_syn_tv = float(getattr(self.opt, "lambda_syn_tv", 0.25)) * self._tv_loss(self.delta_od * dr)
            self.loss_G_syn_mag = (float(getattr(self.opt, "lambda_syn_mag", 0.07))
                                   * torch.sum(torch.abs(self.delta_od) * dr) / reg_den)
            lam_mm = float(getattr(self.opt, "lambda_syn_mask_mean", 0.0))
            if lam_mm > 0:
                self.loss_G_syn_mask_mean = lam_mm * torch.relu(
                    0.05 - torch.sum(torch.abs(self.delta_od) * dr) / reg_den)

        self.loss_G = (
            self.loss_G_GAN + self.loss_G_L1
            + self.loss_G_delta_bg + self.loss_G_delta
            + self.loss_G_grad + self.loss_G_delta_grad
            + self.loss_G_lap + self.loss_G_ssim + self.loss_G_stats
            + self.loss_G_syn_tv + self.loss_G_syn_mag + self.loss_G_syn_mask_mean
        )
        self.loss_G.backward()

    # ─────────────────────────────────────────────────────────────────────────
    # optimize_parameters — with d_update_ratio support
    # ─────────────────────────────────────────────────────────────────────────

    def optimize_parameters(self):
        self.forward()
        self._g_step += 1

        d_ratio = int(getattr(self.opt, "d_update_ratio", 1))
        update_D = (self._g_step % max(1, d_ratio) == 0)

        # D update
        self.set_requires_grad(self.netD, True)
        if update_D:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
        else:
            z = torch.tensor(0.0, device=self.device)
            self.loss_D_fake = self.loss_D_real = self.loss_D = z.clone()

        # G update
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _load_pretrained_netG(self, pretrained_path: str):
        """Load old checkpoint with shape mismatch tolerance (first conv channel expansion)."""
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
            # First conv: expand input channels (e.g. 8→9)
            if "model.model.0.weight" in k and v.ndim == 4 and net_state[k].ndim == 4:
                nw = net_state[k].clone()
                c_old = v.shape[1]
                c_new = nw.shape[1]
                if nw.shape[0] == v.shape[0] and nw.shape[2:] == v.shape[2:] and c_old < c_new:
                    nw[:, :c_old] = v
                    nw[:, c_old:] = 0.0
                    new_state[k] = nw
                    print(f"[pretrain] expanded first conv {v.shape} → {nw.shape}")
                    continue
            print(f"[pretrain] skipped: {k} {v.shape} vs {net_state[k].shape}")
        missing, unexpected = self.netG.load_state_dict(new_state, strict=False)
        print(f"[pretrain] loaded from {pretrained_path} | missing={len(missing)} unexpected={len(unexpected)}")

    def get_current_visuals(self):
        visuals = {}
        for name in ["real_A", "fake_B", "real_B", "empty_E", "tray_T"]:
            attr = {"empty_E": "empty_E", "tray_T": "tray_T"}.get(name, name)
            val = getattr(self, attr, None)
            if val is not None:
                visuals[name] = val
        return visuals