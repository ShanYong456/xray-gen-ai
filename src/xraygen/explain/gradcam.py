import torch
import torch.nn.functional as F

class GradCAM:
    """
    Simple Grad-CAM implementation for PyTorch models.

    Usage:
        gradcam = GradCAM(model, target_layer=model.conv2)
        heatmap = gradcam(input_tensor)  # input_tensor: [1, C, H, W]
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        
        self.activations = None
        self.gradients = None

        # Register hooks
        def fwd_hook(module, inp, out):
            # out: [B, C, H, W]
            self.activations = out

        def bwd_hook(module, grad_in, grad_out):
            # grad_out[0]: [B, C, H, W]
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_backward_hook(bwd_hook)

    def __call__(self, x, target_class=None):
        """
        Args:
            x: input tensor of shape [1, C, H, W]
            target_class: integer class index to explain. If None, uses argmax.

        Returns:
            cam: heatmap as a CPU numpy array of shape [H, W], values in [0, 1]
        """
        self.model.zero_grad()

        # Forward pass
        outputs = self.model(x)
        if target_class is None:
            target_class = outputs.argmax(dim=1).item()

        loss = outputs[0, target_class]
        loss.backward()

        # gradients: [c, H', W']
        grads = self.gradients[0]
        # activiations: [C, H', W']
        acts = self.activations[0]

        # Global-average-pool the gradients to get weights: [C, 1, 1]
        weights = grads.mean(dim=(1, 2), keepdim=True)

        # Weighted sum of activations: [H', W']
        cam = (weights * acts).sum(dim=0)

        # ReLU: Keep only postive influence
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        cam = cam - cam_min
        cam = cam / (cam_max - cam_min + 1e-8)

        # Upsample to input size: [1, 1, H, W]
        cam = cam.unsqueeze(0).unsqueeze(0) # [1, 1, H', W']
        cam = F.interpolate(
            cam,
            size = (x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().detach().cpu().numpy() # [H, W]

        return cam
