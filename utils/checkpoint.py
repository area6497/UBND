
from pathlib import Path
from typing import Any
import torch

def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=0, best_metric=None, extra=None):

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state = {'epoch': epoch, 'model': model.state_dict(), 'best_metric': best_metric, 'extra': extra or {}}
    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
    torch.save(state, output_path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location='cpu', strict=True):

    checkpoint = torch.load(path, map_location=map_location)
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    cleaned = {key.replace('module.', ''): value for key, value in state_dict.items()}
    model.load_state_dict(cleaned, strict=strict)
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    return checkpoint

def latest_checkpoint(checkpoint_dir):

    directory = Path(checkpoint_dir)
    checkpoints = sorted(directory.glob('*.pth'), key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None
