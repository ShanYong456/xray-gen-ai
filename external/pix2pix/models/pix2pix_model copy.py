import torch
import torch.nn.functional as F

from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    """
    pix2pix model with OPTIONAL physics-style delta compositing on an empty tray E.

    Fixes:
      1) Masked GAN background blending uses a realistic reference (empty tray E) to avoid dull/black bias.
      2) Post-physics learnable display mapper to correct "global tone/contrast" without cheating in OD space.

    IMPORTANT pix2pix framework rule:
      - Any network listed in self.model_names MUST be stored as self.net<name>.
        Example: name "display_mapper" => attribute MUST be self.netdisplay_mapper
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm="batch", netG="unet_256", dataset_mode="aligned")
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="vanilla")
            parser.add_argument("--lambda_L1", type=float, default=100.0)

        def _has_dest(dest: str) -> bool:
            return any(getattr(a, "dest", None) == dest for a in parser._actions)

        # ---- delta-compositing ----
        if not _has_dest("use_delta_comp"):
            parser.add_argument("--use_delta_comp", action="store_true")
        if not _has_dest("compose_eps"):
            parser.add_argument("--compose_eps", type=float, default=1e-6)
        if not _has_dest("delta_scale"):
            parser.add_argument("--delta_scale", type=float, default=1.0)
        if not _has_dest("od_gain"):
            parser.add_argument("--od_gain", type=float, default=4.0)

        # NOTE: default ON to avoid "linear looks dull" when your dataset is gamma-encoded images.
        if not _has_dest("use_gamma"):
            parser.add_argument("--use_gamma", action="store_true")
            parser.set_defaults(use_gamma=True)
        if not _has_dest("gamma"):
            parser.add_argument("--gamma", type=float, default=2.2)

        if not _has_dest("delta_nonneg"):
            parser.add_argument("--delta_nonneg", action="store_true")

        # ---- masked L1 ----
        if not _has_dest("use_masked_l1"):
            parser.add_argument("--use_masked_l1", action="store_true")
        if not _has_dest("lambda_bg"):
            parser.add_argument("--lambda_bg", type=float, default=1.5)

        # ---- masked GAN ----
        if not _has_dest("use_masked_gan"):
            parser.add_argument("--use_masked_gan", action="store_true")
        if not _has_dest("gan_bg_keep"):
            parser.add_argument("--gan_bg_keep", type=float, default=0.2)

        if not _has_dest("gan_mask_dilate"):
            parser.add_argument("--gan_mask_dilate", action="store_true")
        if not _has_dest("gan_dilate_px"):
            parser.add_argument("--gan_dilate_px", type=int, default=7)

        if not _has_dest("gan_bg_ref"):
            parser.add_argument(
                "--gan_bg_ref",
                type=str,
                default="empty",
                choices=["empty", "real", "zero"],
                help="Background reference used for masked GAN blending. "
                     "'empty' uses E (recommended). 'real' uses each image itself. 'zero' uses 0.",
            )

        # ---- feature matching ----
        if not _has_dest("use_fm"):
            parser.add_argument("--use_fm", action="store_true")
        if not _has_dest("lambda_fm"):
            parser.add_argument("--lambda_fm", type=float, default=10.0)

        # ---- separate lr ----
        if not _has_dest("lr_G"):
            parser.add_argument("--lr_G", type=float, default=None)
        if not _has_dest("lr_D"):
            parser.add_argument("--lr_D", type=float, default=None)

        # ---- robust mask ----
        if not _has_dest("mask_thr"):
            parser.add_argument("--mask_thr", type=float, default=0.05)

        # ---- display mapper toggle ----
        if not _has_dest("use_display_mapper"):
            parser.add_argument(
                "--use_display_mapper",
                action="store_true",
                help="Enable post-physics display mapper (recommended).",
            )
            parser.set_defaults(use_display_mapper=True)

        # ---- NEW: residual mapper strength (start small) ----
        if not _has_dest("dm_alpha"):
            parser.add_argument("--dm_alpha", type=float, default=0.10,
                                help="Residual strength for display mapper: y = x + alpha * mapper(x).")

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ["G_GAN", "G_L1", "D_real", "D_fake"]
        if self.isTrain and getattr(opt, "use_fm", False):
            self.loss_names += ["G_FM"]

        self.visual_names = ["real_A", "fake_B", "real_B"]
        self.model_names = ["G", "D"] if self.isTrain else ["G"]

        # ---- networks ----
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

        # ---- display mapper (post-physics) ----
        self.use_display_mapper = bool(getattr(opt, "use_display_mapper", True))
        self.dm_alpha = float(getattr(opt, "dm_alpha", 0.10))

        if self.use_display_mapper:
            # Residual mapper (NO tanh; we residual-add and clamp in [-1,1])
            self.netdisplay_mapper = torch.nn.Sequential(
                torch.nn.Conv2d(opt.output_nc, 32, kernel_size=1, stride=1, padding=0),
                torch.nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
                torch.nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(32, opt.output_nc, kernel_size=1, stride=1, padding=0),
            ).to(self.device)

            # Initialize last conv to near-zero so mapper starts as identity (residual ~0)
            with torch.no_grad():
                last = self.netdisplay_mapper[-1]
                if isinstance(last, torch.nn.Conv2d):
                    torch.nn.init.zeros_(last.weight)
                    if last.bias is not None:
                        torch.nn.init.zeros_(last.bias)

            if "display_mapper" not in self.model_names:
                self.model_names.append("display_mapper")
        else:
            self.netdisplay_mapper = None

        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            lr_G = opt.lr if getattr(opt, "lr_G", None) is None else float(opt.lr_G)
            lr_D = opt.lr if getattr(opt, "lr_D", None) is None else float(opt.lr_D)

            # IMPORTANT: optimizer_G includes display mapper params
            g_params = list(self.netG.parameters())
            if self.use_display_mapper and (self.netdisplay_mapper is not None):
                g_params += list(self.netdisplay_mapper.parameters())

            self.optimizer_G = torch.optim.Adam(g_params, lr=lr_G, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr_D, betas=(opt.beta1, 0.999))
            self.optimizers += [self.optimizer_G, self.optimizer_D]

        # holders
        self.empty_E = None
        self.delta = None
        self.mask_M = None

    # -----------------------------
    # data / masks
    # -----------------------------
    def set_input(self, input):
        AtoB = self.opt.direction == "AtoB"
        A = input["A" if AtoB else "B"].to(self.device)
        B = input["B" if AtoB else "A"].to(self.device)
        self.real_B = B

        if getattr(self.opt, "use_delta_comp", False):
            if "E" not in input:
                raise KeyError("use_delta_comp is enabled but dataset did not return 'E'.")
            self.empty_E = input["E"].to(self.device)
            self.real_A = torch.cat([A, self.empty_E], dim=1)
        else:
            self.real_A = A
            self.empty_E = None

        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def _build_object_mask_from_A(self):
        A_only = self.real_A[:, :3, :, :]
        thr = float(getattr(self.opt, "mask_thr", 0.05))
        M = (A_only.abs().sum(dim=1, keepdim=True) > thr).float()
        return A_only, M

    def _dilate_mask(self, M: torch.Tensor, radius_px: int) -> torch.Tensor:
        r = int(max(0, radius_px))
        if r == 0:
            return M
        k = 2 * r + 1
        return F.max_pool2d(M, kernel_size=k, stride=1, padding=r)

    def _get_gan_mask(self, M: torch.Tensor) -> torch.Tensor:
        if bool(getattr(self.opt, "use_masked_gan", False)) and bool(getattr(self.opt, "gan_mask_dilate", False)):
            return self._dilate_mask(M, int(getattr(self.opt, "gan_dilate_px", 7)))
        return M

    # -----------------------------
    # masked GAN blending
    # -----------------------------
    def _choose_bg_ref(self, img: torch.Tensor) -> torch.Tensor:
        ref = getattr(self.opt, "gan_bg_ref", "empty")
        if ref == "real":
            return img
        if ref == "zero":
            return torch.zeros_like(img)
        # empty
        if self.empty_E is not None and self.empty_E.shape == img.shape:
            return self.empty_E
        return torch.zeros_like(img)

    def _blend_mask_for_gan(self, img: torch.Tensor, M: torch.Tensor, bg_keep: float) -> torch.Tensor:
        bg_keep = float(bg_keep)
        bg_keep = 0.0 if bg_keep < 0.0 else (1.0 if bg_keep > 1.0 else bg_keep)

        M3 = M.expand_as(img)
        bg_ref = self._choose_bg_ref(img)

        # object stays from img, background becomes mix(img, bg_ref)
        bg = bg_keep * img + (1.0 - bg_keep) * bg_ref
        return img * M3 + bg * (1.0 - M3)

    # -----------------------------
    # helpers
    # -----------------------------
    def _netD_forward_with_features(self, x):
        if not hasattr(self.netD, "model"):
            pred = self.netD(x)
            return pred, []
        feats = []
        h = x
        layers = list(self.netD.model)
        for i, layer in enumerate(layers):
            h = layer(h)
            if i < len(layers) - 1:
                feats.append(h)
        return h, feats

    def _to_01(self, x):
        return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)

    def _to_m11(self, x01):
        return x01 * 2.0 - 1.0

    def _gamma_decode(self, x01):
        # decode from gamma-encoded to linear
        g = float(getattr(self.opt, "gamma", 2.2))
        return torch.clamp(x01, 0.0, 1.0).pow(g)

    def _gamma_encode(self, xlin):
        # encode from linear to gamma-encoded
        g = float(getattr(self.opt, "gamma", 2.2))
        invg = 1.0 / g if g > 0 else 1.0
        return torch.clamp(xlin, 0.0, 1.0).pow(invg)

    def _apply_display_mapper(self, x_m11: torch.Tensor) -> torch.Tensor:
        """
        Residual tone mapper:
            y = clamp(x + alpha * mapper(x), -1, 1)
        mapper starts near 0 due to init, so early training doesn't collapse brightness.
        """
        if not (self.use_display_mapper and (self.netdisplay_mapper is not None)):
            return x_m11
        delta = self.netdisplay_mapper(x_m11)
        y = x_m11 + self.dm_alpha * delta
        return torch.clamp(y, -1.0, 1.0)

    # -----------------------------
    # forward (physics + display)
    # -----------------------------
    def forward(self):
        # non-physics path (regular pix2pix)
        if not getattr(self.opt, "use_delta_comp", False):
            fake = self.netG(self.real_A)
            self.fake_B = self._apply_display_mapper(fake)
            return

        # physics delta prediction
        self.delta = self.netG(self.real_A)

        # object mask from A (first 3 channels)
        _, M = self._build_object_mask_from_A()
        self.mask_M = M

        M_exp = M.expand(-1, self.delta.shape[1], -1, -1) if self.delta.shape[1] != 1 else M

        eps = float(getattr(self.opt, "compose_eps", 1e-6))
        delta_scale = float(getattr(self.opt, "delta_scale", 1.0))
        od_gain = float(getattr(self.opt, "od_gain", 4.0))
        use_gamma = bool(getattr(self.opt, "use_gamma", True))

        if self.empty_E is None:
            raise RuntimeError("use_delta_comp=True but empty_E is None. Dataset must provide 'E'.")

        # empty tray to [0,1]
        E01 = self._to_01(self.empty_E)

        # IMPORTANT: convert to linear before OD, if enabled
        E_lin = self._gamma_decode(E01) if use_gamma else E01

        # optical density
        OD_E = -torch.log(E_lin + eps)

        # delta in OD
        if bool(getattr(self.opt, "delta_nonneg", False)):
            delta_od = F.softplus(self.delta) * (od_gain * delta_scale)
        else:
            delta_od = self.delta * (od_gain * delta_scale)

        # compose in OD space
        OD_pred = OD_E + M_exp * delta_od

        # back to intensity (linear)
        I_lin = torch.exp(-OD_pred)
        I_lin = torch.clamp(I_lin, 0.0, 1.0)

        # IMPORTANT: convert back to gamma-encoded if enabled
        I01 = self._gamma_encode(I_lin) if use_gamma else I_lin

        fake_phys = self._to_m11(I01)

        # post-physics display mapper
        self.fake_B = self._apply_display_mapper(fake_phys)

    # -----------------------------
    # backward D
    # -----------------------------
    def backward_D(self):
        _, M = self._build_object_mask_from_A()
        self.mask_M = M

        use_masked_gan = bool(getattr(self.opt, "use_masked_gan", False))
        bg_keep = float(getattr(self.opt, "gan_bg_keep", 0.2))
        M_gan = self._get_gan_mask(M)

        fake_B_for_D = self.fake_B
        real_B_for_D = self.real_B

        if use_masked_gan:
            fake_B_for_D = self._blend_mask_for_gan(fake_B_for_D, M_gan, bg_keep=bg_keep)
            real_B_for_D = self._blend_mask_for_gan(real_B_for_D, M_gan, bg_keep=bg_keep)

        fake_AB = torch.cat((self.real_A, fake_B_for_D), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat((self.real_A, real_B_for_D), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    # -----------------------------
    # backward G
    # -----------------------------
    def backward_G(self):
        _, M = self._build_object_mask_from_A()
        self.mask_M = M

        use_masked_gan = bool(getattr(self.opt, "use_masked_gan", False))
        bg_keep = float(getattr(self.opt, "gan_bg_keep", 0.2))
        use_fm = bool(getattr(self.opt, "use_fm", False))
        lambda_fm = float(getattr(self.opt, "lambda_fm", 10.0))

        M_gan = self._get_gan_mask(M)

        fake_B_for_D = self.fake_B
        real_B_for_D = self.real_B
        if use_masked_gan:
            fake_B_for_D = self._blend_mask_for_gan(fake_B_for_D, M_gan, bg_keep=bg_keep)
            real_B_for_D = self._blend_mask_for_gan(real_B_for_D, M_gan, bg_keep=bg_keep)

        fake_AB = torch.cat((self.real_A, fake_B_for_D), 1)

        # GAN loss
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # L1 loss
        if getattr(self.opt, "use_delta_comp", False) and getattr(self.opt, "use_masked_l1", False):
            if self.empty_E is None:
                raise RuntimeError("use_masked_l1 requires empty_E (E).")

            eps = float(getattr(self.opt, "compose_eps", 1e-6))
            M_obj = M.expand_as(self.fake_B)

            obj_err = torch.abs(self.fake_B - self.real_B) * M_obj
            obj_l1 = obj_err.sum() / (M_obj.sum() + eps)

            bg_err = torch.abs(self.fake_B - self.empty_E) * (1.0 - M_obj)
            bg_l1 = bg_err.sum() / ((1.0 - M_obj).sum() + eps)

            lambda_bg = float(getattr(self.opt, "lambda_bg", 1.5))
            self.loss_G_L1 = (obj_l1 + lambda_bg * bg_l1) * float(self.opt.lambda_L1)
        else:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * float(self.opt.lambda_L1)

        # feature matching
        self.loss_G_FM = 0.0
        if use_fm:
            real_AB = torch.cat((self.real_A, real_B_for_D), 1)
            _, feats_real = self._netD_forward_with_features(real_AB.detach())
            _, feats_fake = self._netD_forward_with_features(fake_AB)

            if feats_real and feats_fake:
                fm = 0.0
                for fr, ff in zip(feats_real, feats_fake):
                    fm = fm + torch.mean(torch.abs(ff - fr.detach()))
                self.loss_G_FM = fm * lambda_fm

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_FM
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G (and display mapper)
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()