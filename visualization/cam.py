
from pathlib import Path
import cv2
import numpy as np
import torch
CAM_METHODS = {'gradcam': 'GradCAM', 'gradcam_plus_plus': 'GradCAMPlusPlus', 'xgradcam': 'XGradCAM', 'eigengradcam': 'EigenGradCAM', 'layercam': 'LayerCAM'}

def generate_cam(model, input_tensor, rgb_image, output_path, method='gradcam', target_category=None):

    try:
        from pytorch_grad_cam import EigenGradCAM, GradCAM, GradCAMPlusPlus, LayerCAM, XGradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError as exc:
        raise ImportError('Please install grad-cam：pip install grad-cam') from exc
    method_map = {'GradCAM': GradCAM, 'GradCAMPlusPlus': GradCAMPlusPlus, 'XGradCAM': XGradCAM, 'EigenGradCAM': EigenGradCAM, 'LayerCAM': LayerCAM}
    cam_class_name = CAM_METHODS.get(method.lower())
    if cam_class_name is None:
        raise ValueError(f'Unsupported CAM method: {method}')
    target_layer = model.backbone.dual_ib
    targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None
    with method_map[cam_class_name](model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    visualization = show_cam_on_image(rgb_image.astype(np.float32), grayscale_cam, use_rgb=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
