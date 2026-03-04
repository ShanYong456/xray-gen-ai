import torch
from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    """pix2pix model with OPTIONAL physics-style delta compositing on an empty tray E.

    When enabled:
      - dataset must provide input["E"] (empty tray image, same normalization as A/B)
      - generator input becomes concat([A, E]) so input_nc must match (e.g. 3+3=6)
      - generator predicts delta (same channels as B)
      - model composes fake_B internally in log space (optical density)

    When disabled (default), behaves like standard pix2pix.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add model-specific options and set defaults."""
        parser.set_defaults(norm="batch", netG="unet_256", dataset_mode="aligned")
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="vanilla")
            parser.add_argument("--lambda_L1", type=float, default=100.0, help="weight for L1 loss")

        # ---- NEW options for delta-compositing (physics) ----
        
        parser.add_argument(
            "--compose_eps", type=float, default=1e-6,
            help="epsilon for log/exp stability in delta composition"
        )
        parser.add_argument(
            "--delta_scale", type=float, default=1.0,
            help="scale applied to predicted delta before adding in log-space"
        )
        parser.add_argument(
            "--use_masked_l1", action="store_true",
            help="If set (and use_delta_comp), use masked object L1 + background identity L1."
        )
        parser.add_argument(
            "--lambda_bg", type=float, default=1.5,
            help="background identity strength vs empty tray outside mask (only used with use_masked_l1)"
        )

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class."""
        BaseModel.__init__(self, opt)

        self.loss_names = ["G_GAN", "G_L1", "D_real", "D_fake"]
        self.visual_names = ["real_A", "fake_B", "real_B"]

        if self.isTrain:
            self.model_names = ["G", "D"]
        else:
            self.model_names = ["G"]

        self.device = opt.device

        # define networks
        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
            not opt.no_dropout, opt.init_type, opt.init_gain
        )

        if self.isTrain:
            self.netD = networks.define_D(
                opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain
            )

        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        # holders for physics mode
        self.empty_E = None
        self.delta = None
        self.mask_M = None

    def set_input(self, input):
        """Unpack input data from the dataloader and perform pre-processing steps.

        Standard pix2pix:
          real_A = A (or B) depending on direction
          real_B = B (or A)

        Delta-compositing pix2pix:
          expects input["E"] = empty tray (same normalization as B)
          real_A becomes concat([A, E]) and empty_E is stored separately.
        """
        AtoB = self.opt.direction == "AtoB"

        A = input["A" if AtoB else "B"].to(self.device)
        B = input["B" if AtoB else "A"].to(self.device)
        self.real_B = B

        if getattr(self.opt, "use_delta_comp", False):
            if "E" not in input:
                raise KeyError(
                    "use_delta_comp is enabled but input dict has no key 'E'. "
                    "Your dataset must return {'E': empty_tray_tensor}."
                )
            E = input["E"].to(self.device)
            self.empty_E = E
            # condition = [A, E]
            self.real_A = torch.cat([A, E], dim=1)
        else:
            self.real_A = A
            self.empty_E = None

        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def _build_object_mask_from_A(self):
        """Build object mask M from the FIRST 3 channels of the condition (assumes mask is 3ch color)."""
        # If your mask has different channels, adjust this slice.
        A_only = self.real_A[:, :3, :, :]
        M = (A_only.abs().sum(dim=1, keepdim=True) > 0).float()  # (N,1,H,W)
        return A_only, M

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if not getattr(self.opt, "use_delta_comp", False):
            self.fake_B = self.netG(self.real_A)  # G(A)
            return

        # ---- netG predicts delta ----
        self.delta = self.netG(self.real_A)  # predicted residual (same channels as B)

        # ---- object mask from A ----
        _, M = self._build_object_mask_from_A()
        self.mask_M = M

        # expand mask to channels
        if self.delta.shape[1] != 1:
            M_exp = M.expand(-1, self.delta.shape[1], -1, -1)
        else:
            M_exp = M

        eps = float(getattr(self.opt, "compose_eps", 1e-6))
        delta_scale = float(getattr(self.opt, "delta_scale", 1.0))

        # Convert empty tray to [0,1] intensity
        E01 = (self.empty_E + 1.0) * 0.5
        E01 = torch.clamp(E01, 0.0, 1.0)

        # Optical density: OD = -log(I)
        OD_E = -torch.log(E01 + eps)

        # delta in OD space (allow +/- by default)
        delta_od = self.delta * delta_scale

        # Compose: OD_pred = OD_E + M * delta_od
        OD_pred = OD_E + M_exp * delta_od

        # Back to intensity: I = exp(-OD)
        I_pred = torch.exp(-OD_pred)
        I_pred = torch.clamp(I_pred, 0.0, 1.0)

        # Back to [-1,1]
        self.fake_B = I_pred * 2.0 - 1.0

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # L1 loss
        if getattr(self.opt, "use_delta_comp", False) and getattr(self.opt, "use_masked_l1", False):
            # object mask
            _, M_obj = self._build_object_mask_from_A()  # (N,1,H,W)
            M_obj = M_obj.expand_as(self.fake_B)

            # object region matches real_B
            obj_l1 = (torch.abs(self.fake_B - self.real_B) * M_obj).mean()

            # background region stays like empty_E
            bg_l1 = (torch.abs(self.fake_B - self.empty_E) * (1.0 - M_obj)).mean()

            lambda_bg = float(getattr(self.opt, "lambda_bg", 1.5))
            self.loss_G_L1 = (obj_l1 + lambda_bg * bg_l1) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()