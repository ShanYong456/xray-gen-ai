import torch
import torch.nn.functional as F

from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    """
    Structured-condition pix2pix without empty-tray / delta compositing.

    Input:
        A = conditioning tensor
            [mask, edge?, thickness?, coord_x?, coord_y?, appearance?]

    Target:
        B = target image

    Notes:
    - Supports both real and synthetic samples.
    - GAN loss is used for both real and synthetic samples, with reduced
      weight on synthetic samples via syn_gan_weight.
    - Optional masked L1, gradient, Laplacian, SSIM, and region-stat losses.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm="instance", netG="unet_256", dataset_mode="aligned")

        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="lsgan")
            parser.add_argument("--lambda_L1", type=float, default=30.0)

        # Conditioning layout
        parser.add_argument("--class_nc", type=int, default=1)
        parser.add_argument("--use_appearance_channel", action="store_true")
        parser.add_argument("--appearance_nc", type=int, default=1)
        parser.add_argument("--use_thickness_channel", action="store_true")
        parser.add_argument("--thickness_nc", type=int, default=1)
        parser.add_argument("--mask_thr", type=float, default=0.05)

        # Masked L1
        parser.add_argument("--use_masked_l1", action="store_true")
        parser.add_argument("--lambda_bg", type=float, default=1.5)

        # Tray
        parser.add_argument("--use_tray_mask", action="store_true")

        # Detail losses
        parser.add_argument("--use_grad_loss", action="store_true")
        parser.add_argument("--lambda_grad", type=float, default=10.0)
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

        # GAN stability
        parser.add_argument("--lambda_gp", type=float, default=0.0)
        parser.add_argument("--d_label_smooth", type=float, default=0.1)
        parser.add_argument("--syn_gan_weight", type=float, default=0.3)
        parser.add_argument("--d_update_ratio", type=int, default=1)

        parser.add_argument("--pretrained_netG", type=str, default="")
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = [
            "G_GAN", "G_L1",
            "G_grad", "G_lap", "G_ssim", "G_stats",
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
        self._g_step = 0

        self.quality_ema = None
        self.quality_best = -1e9
        self.quality_last = None
        self.metric_momentum = 0.95
        self.loss_names += ["Q_score", "Q_ema", "Q_best", "Q_trend"]
        for name in ["Q_score", "Q_ema", "Q_best", "Q_trend"]:
            setattr(self, f"loss_{name}", torch.tensor(0.0, device=self.device))

        self.netG = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
        )

        pretrained = str(getattr(opt, "pretrained_netG", "")).strip()
        if pretrained:
            self._load_pretrained_netG(pretrained)

        if self.isTrain:
            self.netD = networks.define_D(
                opt.input_nc + opt.output_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.norm,
                opt.init_type,
                opt.init_gain,
            )
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers += [self.optimizer_G, self.optimizer_D]

        # State tensors
        self.real_A = None
        self.real_B = None
        self.fake_B = None
        self.mask_M = None
        self.tray_T = None
        self.instance_masks = None
        self.is_synthetic = False
        self.has_real_B = False

        z = torch.tensor(0.0, device=self.device)
        for name in ["G_grad", "G_lap", "G_ssim", "G_stats"]:
            setattr(self, f"loss_{name}", z.clone())

        self._init_kernels()

    # ─────────────────────────────────────────────────────────────────────────
    # Kernel setup
    # ─────────────────────────────────────────────────────────────────────────

    def _init_kernels(self):
        kx = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 4.0
        ky = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 4.0
        kl = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.sobel_kx = kx.to(self.device)
        self.sobel_ky = ky.to(self.device)
        self.lap_k = kl.to(self.device)

    def _gaussian_blur(self, x, ksize=5, sigma=1.2):
        if ksize <= 1:
            return x
        ksize = ksize + (0 if ksize % 2 else 1)
        C = x.shape[1]

        g = torch.exp(
            -(torch.arange(ksize, device=x.device) - (ksize - 1) / 2) ** 2 / (2 * sigma ** 2)
        )
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
                        A[:, s:e] + torch.randn_like(A[:, s:e]) * noise_std, -1, 1
                    )

        is_synth = input.get("is_synthetic", False)
        self.is_synthetic = bool(
            is_synth.flatten()[0].item() if torch.is_tensor(is_synth) else is_synth
        )

        b_key = "B" if AtoB else "A"
        if b_key in input and input[b_key] is not None:
            self.real_B = input[b_key].to(self.device)
            self.has_real_B = True
        else:
            self.real_B = None
            self.has_real_B = False

        self.real_A = A

        self.tray_T = (
            input["T"].to(self.device)
            if getattr(self.opt, "use_tray_mask", False) and "T" in input
            else None
        )

        inst = input.get("instance_masks", None)
        self.instance_masks = inst.to(self.device) if inst is not None else None
        self.image_paths = input.get("A_paths" if AtoB else "B_paths", [])

    # ─────────────────────────────────────────────────────────────────────────
    # Mask / conditioning helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _appearance_channel_start_idx(self):
        start = self.class_nc
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
                0.0,
                1.0,
            )
            blur_k = int(getattr(self.opt, "mask_blur_ksize", 0))
            if blur_k > 1:
                M = self._gaussian_blur(
                    M,
                    blur_k,
                    float(getattr(self.opt, "mask_blur_sigma", 1.2))
                )
            return None, torch.clamp(M, 0.0, 1.0).float()

        # NEW: aggregate first class_nc channels instead of assuming 1 object channel
        A01 = (self.real_A[:, :self.class_nc] + 1.0) * 0.5
        A01 = torch.clamp(A01, 0.0, 1.0)

        if self.class_nc > 1:
            A_obj = torch.clamp(torch.sum(A01, dim=1, keepdim=True), 0.0, 1.0)
        else:
            A_obj = A01[:, :1]

        thr = float(getattr(self.opt, "mask_thr", 0.05))
        if getattr(self.opt, "use_soft_mask", False):
            M = torch.sigmoid(
                float(getattr(self.opt, "mask_soft_beta", 30.0)) * (A_obj - thr)
            )
        else:
            M = (A_obj > thr).float()

        blur_k = int(getattr(self.opt, "mask_blur_ksize", 0))
        if blur_k > 1:
            M = self._gaussian_blur(
                M,
                blur_k,
                float(getattr(self.opt, "mask_blur_sigma", 1.2))
            )

        return A_obj, torch.clamp(M, 0.0, 1.0)

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
        return torch.sqrt(
            F.conv2d(x, kx, padding=1, groups=C) ** 2 +
            F.conv2d(x, ky, padding=1, groups=C) ** 2 +
            1e-12
        )

    def _laplacian(self, x):
        C = x.shape[1]
        return F.conv2d(
            x,
            self.lap_k.to(x.dtype).repeat(C, 1, 1, 1),
            padding=1,
            groups=C,
        )

    def _ssim_map(self, x, y, window_size=7):
        pad = window_size // 2
        mu_x = F.avg_pool2d(x, window_size, stride=1, padding=pad)
        mu_y = F.avg_pool2d(y, window_size, stride=1, padding=pad)
        sx = F.avg_pool2d(x * x, window_size, stride=1, padding=pad) - mu_x * mu_x
        sy = F.avg_pool2d(y * y, window_size, stride=1, padding=pad) - mu_y * mu_y
        sxy = F.avg_pool2d(x * y, window_size, stride=1, padding=pad) - mu_x * mu_y
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        return ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / (
            (mu_x ** 2 + mu_y ** 2 + C1) * (sx + sy + C2) + 1e-12
        )

    # ─────────────────────────────────────────────────────────────────────────
    # forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self):
        self.fake_B = self.netG(self.real_A)
        _, M = self._build_object_mask_from_A()
        if getattr(self.opt, "use_tray_mask", False) and self.tray_T is not None:
            M = M * self.tray_T
        self.mask_M = M

    # ─────────────────────────────────────────────────────────────────────────
    # Gradient penalty (R1)
    # ─────────────────────────────────────────────────────────────────────────

    def _r1_penalty(self, real_AB):
        real_AB = real_AB.detach().requires_grad_(True)
        pred_real = self.netD(real_AB)

        # handle multiscale / nested outputs
        if isinstance(pred_real, list):
            pred_real_last = pred_real[-1]
            if isinstance(pred_real_last, list):
                pred_real_last = pred_real_last[-1]
            pred_real = pred_real_last

        grad = torch.autograd.grad(
            outputs=pred_real.sum(),
            inputs=real_AB,
            create_graph=True,
            retain_graph=True,
        )[0]
        return grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()

    # ─────────────────────────────────────────────────────────────────────────
    # backward_D
    # ─────────────────────────────────────────────────────────────────────────

    def backward_D(self):
        smooth = float(getattr(self.opt, "d_label_smooth", 0.1))
        lam_gp = float(getattr(self.opt, "lambda_gp", 0.0))

        fake_AB = torch.cat((self.real_A, self.fake_B.detach()), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        if self.has_real_B:
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB)

            if smooth > 0:
                if isinstance(pred_real, list):
                    losses = []
                    for p in pred_real:
                        p_use = p[-1] if isinstance(p, list) else p
                        target = torch.ones_like(p_use) * (1.0 - smooth)
                        losses.append(F.mse_loss(p_use, target))
                    self.loss_D_real = sum(losses) / len(losses)
                else:
                    target = torch.ones_like(pred_real) * (1.0 - smooth)
                    self.loss_D_real = F.mse_loss(pred_real, target)
            else:
                self.loss_D_real = self.criterionGAN(pred_real, True)

            gp_loss = 0.0
            if lam_gp > 0:
                gp_loss = self._r1_penalty(real_AB) * lam_gp

            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 + gp_loss
        else:
            self.loss_D_real = torch.tensor(0.0, device=self.device)
            self.loss_D = self.loss_D_fake * 0.5

        self.loss_D.backward()

    # ─────────────────────────────────────────────────────────────────────────
    # backward_G
    # ─────────────────────────────────────────────────────────────────────────

    def backward_G(self):
        z = torch.tensor(0.0, device=self.device)
        syn_weight = float(getattr(self.opt, "syn_gan_weight", 0.3))

        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        gan_raw = self.criterionGAN(pred_fake, True)
        gan_scale = 1.0 if (not self.is_synthetic and self.has_real_B) else syn_weight
        self.loss_G_GAN = gan_raw * gan_scale

        self.loss_G_grad = z.clone()
        self.loss_G_lap = z.clone()
        self.loss_G_ssim = z.clone()
        self.loss_G_stats = z.clone()

        _, M_obj_1 = self._build_object_mask_from_A()
        if getattr(self.opt, "use_tray_mask", False) and self.tray_T is not None:
            M_obj_1 = M_obj_1 * self.tray_T

        M_obj = self._expand_like(M_obj_1, self.fake_B)
        T = self._expand_like(self.tray_T, self.fake_B) if self.tray_T is not None else torch.ones_like(M_obj)
        region = M_obj * T
        eps = 1e-6

        # L1
        if self.has_real_B:
            if getattr(self.opt, "use_masked_l1", False):
                obj_r = region
                bg_r = (1.0 - M_obj) * T

                obj_l1 = torch.sum(torch.abs(self.fake_B - self.real_B) * obj_r) / (torch.sum(obj_r) + eps)
                bg_l1 = torch.sum(torch.abs(self.fake_B - self.real_B) * bg_r) / (torch.sum(bg_r) + eps)

                self.loss_G_L1 = (
                    obj_l1 + float(getattr(self.opt, "lambda_bg", 1.5)) * bg_l1
                ) * self.opt.lambda_L1
            else:
                self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = z.clone()

        # Gradient loss
        if self.has_real_B and getattr(self.opt, "use_grad_loss", False):
            lam = float(getattr(self.opt, "lambda_grad", 10.0))
            num = torch.sum(
                torch.abs(self._sobel_mag(self.fake_B) - self._sobel_mag(self.real_B)) * region
            )
            self.loss_G_grad = lam * num / (torch.sum(region) + eps)

        # Laplacian loss
        if self.has_real_B and getattr(self.opt, "use_lap_loss", False):
            lam = float(getattr(self.opt, "lambda_lap", 6.0))
            num = torch.sum(
                torch.abs(self._laplacian(self.fake_B) - self._laplacian(self.real_B)) * region
            )
            self.loss_G_lap = lam * num / (torch.sum(region) + eps)

        # SSIM loss
        if self.has_real_B and getattr(self.opt, "use_ssim_loss", False):
            lam = float(getattr(self.opt, "lambda_ssim", 3.0))
            f01 = torch.clamp((self.fake_B + 1.0) * 0.5, 0.0, 1.0)
            r01 = torch.clamp((self.real_B + 1.0) * 0.5, 0.0, 1.0)
            num = torch.sum((1.0 - self._ssim_map(f01, r01)) * region)
            self.loss_G_ssim = lam * num / (torch.sum(region) + eps)

        # Region stats
        if self.has_real_B and getattr(self.opt, "use_region_stats", False):
            lam = float(getattr(self.opt, "lambda_stats", 3.0))
            f01 = torch.clamp((self.fake_B + 1.0) * 0.5, 0.0, 1.0)
            r01 = torch.clamp((self.real_B + 1.0) * 0.5, 0.0, 1.0)

            reg_sum = torch.sum(region, dim=(2, 3), keepdim=True) + eps
            f_mean = torch.sum(f01 * region, dim=(2, 3), keepdim=True) / reg_sum
            r_mean = torch.sum(r01 * region, dim=(2, 3), keepdim=True) / reg_sum

            f_std = torch.sqrt(
                torch.sum((f01 - f_mean) ** 2 * region, dim=(2, 3), keepdim=True) / reg_sum + 1e-12
            )
            r_std = torch.sqrt(
                torch.sum((r01 - r_mean) ** 2 * region, dim=(2, 3), keepdim=True) / reg_sum + 1e-12
            )

            self.loss_G_stats = lam * (
                torch.mean(torch.abs(f_mean - r_mean)) +
                torch.mean(torch.abs(f_std - r_std))
            )

        self.loss_G = (
            self.loss_G_GAN +
            self.loss_G_L1 +
            self.loss_G_grad +
            self.loss_G_lap +
            self.loss_G_ssim +
            self.loss_G_stats
        )
        self._compute_quality_score()
        self.loss_G.backward()

    # ─────────────────────────────────────────────────────────────────────────
    # optimize_parameters
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
            self.loss_D_fake = z.clone()
            self.loss_D_real = z.clone()
            self.loss_D = z.clone()

        # G update
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _load_pretrained_netG(self, pretrained_path: str):
        """
        Load old class_nc=1 checkpoint into new class_nc=2 generator.

        Old Stage8 layout:
        input_nc = 5
        [obj, edge, thickness, coord_x, coord_y]

        New Stage10 layout:
        input_nc = 6
        [shampoo, tray, edge, thickness, coord_x, coord_y]

        Mapping:
        old obj      -> new shampoo
        old obj      -> new tray
        old edge     -> new edge
        old thickness-> new thickness
        old coord_x  -> new coord_x
        old coord_y  -> new coord_y
        """
        state = torch.load(pretrained_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        net_state = self.netG.state_dict()
        new_state = {}

        # Find first conv key robustly
        first_conv_key = None
        for k, v in net_state.items():
            if v.ndim == 4:
                first_conv_key = k
                break

        for k, v in state.items():
            if k not in net_state:
                continue

            if net_state[k].shape == v.shape:
                new_state[k] = v
                continue

            if k == first_conv_key and v.ndim == 4 and net_state[k].ndim == 4:
                dst = net_state[k].clone()
                dst.zero_()

                out_c_new, in_c_new, kh_new, kw_new = dst.shape
                out_c_old, in_c_old, kh_old, kw_old = v.shape

                if out_c_new != out_c_old or kh_new != kh_old or kw_new != kw_old:
                    print(f"[pretrain] skip first conv due to incompatible shape: "
                        f"{k} old={tuple(v.shape)} new={tuple(dst.shape)}")
                    continue

                # Exact intended mapping: old 5ch -> new 6ch
                if in_c_old == 5 and in_c_new == 6:
                    # old obj -> shampoo
                    dst[:, 0:1, :, :] = v[:, 0:1, :, :]

                    # old obj -> tray
                    dst[:, 1:2, :, :] = v[:, 0:1, :, :]

                    # old edge -> new edge
                    dst[:, 2:3, :, :] = v[:, 1:2, :, :]

                    # old thickness -> new thickness
                    dst[:, 3:4, :, :] = v[:, 2:3, :, :]

                    # old coord_x -> new coord_x
                    dst[:, 4:5, :, :] = v[:, 3:4, :, :]

                    # old coord_y -> new coord_y
                    dst[:, 5:6, :, :] = v[:, 4:5, :, :]

                    new_state[k] = dst
                    print(f"[pretrain] adapted first conv 5->6: {k} old={tuple(v.shape)} new={tuple(dst.shape)}")
                    continue

                # Generic fallback: overlapping prefix copy
                copy_in = min(in_c_old, in_c_new)
                dst[:, :copy_in, :, :] = v[:, :copy_in, :, :]
                if in_c_new > copy_in:
                    torch.nn.init.normal_(dst[:, copy_in:, :, :], mean=0.0, std=0.02)
                new_state[k] = dst
                print(f"[pretrain] prefix-copied first conv {k}: old={tuple(v.shape)} new={tuple(dst.shape)}")
                continue

            print(f"[pretrain] skipped: {k} {tuple(v.shape)} vs {tuple(net_state[k].shape)}")

        missing, unexpected = self.netG.load_state_dict(new_state, strict=False)
        print(
            f"[pretrain] loaded from {pretrained_path} | "
            f"loaded={len(new_state)} missing={len(missing)} unexpected={len(unexpected)}"
        )

    def get_current_visuals(self):
        visuals = {}
        for name in ["real_A", "fake_B", "real_B", "tray_T"]:
            val = getattr(self, name, None)
            if val is not None:
                visuals[name] = val
        return visuals
    

    def get_current_quality_summary(self):
        return {
            "score": self._safe_float(getattr(self, "loss_Q_score", 0.0)),
            "ema": self._safe_float(getattr(self, "loss_Q_ema", 0.0)),
            "best": self._safe_float(getattr(self, "loss_Q_best", 0.0)),
            "trend": self._safe_float(getattr(self, "loss_Q_trend", 0.0)),
        }

    # METRIC

    def _safe_float(self, x):
        if x is None:
            return 0.0
        if torch.is_tensor(x):
            return float(x.detach().mean().item())
        return float(x)

    def _compute_quality_score(self):
        g_gan   = self._safe_float(getattr(self, "loss_G_GAN", 0.0))
        g_l1    = self._safe_float(getattr(self, "loss_G_L1", 0.0))
        g_grad  = self._safe_float(getattr(self, "loss_G_grad", 0.0))
        g_lap   = self._safe_float(getattr(self, "loss_G_lap", 0.0))
        g_ssim  = self._safe_float(getattr(self, "loss_G_ssim", 0.0))
        g_stats = self._safe_float(getattr(self, "loss_G_stats", 0.0))

        score_gan   = 1.0 / (1.0 + g_gan)
        score_l1    = 1.0 / (1.0 + g_l1)
        score_grad  = 1.0 / (1.0 + g_grad)
        score_lap   = 1.0 / (1.0 + g_lap)
        score_ssim  = 1.0 / (1.0 + g_ssim)
        score_stats = 1.0 / (1.0 + g_stats)

        quality_raw = (
            0.10 * score_gan +
            0.35 * score_l1 +
            0.15 * score_grad +
            0.10 * score_lap +
            0.20 * score_ssim +
            0.10 * score_stats
        )

        quality = 100.0 * quality_raw

        if self.quality_ema is None:
            self.quality_ema = quality
        else:
            m = self.metric_momentum
            self.quality_ema = m * self.quality_ema + (1.0 - m) * quality

        self.quality_best = max(self.quality_best, quality)

        if self.quality_last is None:
            trend = 0.0
        else:
            trend = quality - self.quality_last
        self.quality_last = quality

        self.loss_Q_score = torch.tensor(quality, device=self.device)
        self.loss_Q_ema   = torch.tensor(self.quality_ema, device=self.device)
        self.loss_Q_best  = torch.tensor(self.quality_best, device=self.device)
        self.loss_Q_trend = torch.tensor(trend, device=self.device)