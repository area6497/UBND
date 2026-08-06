
from collections import Counter
from typing import Iterable
import torch
from datasets.breast_ultrasound_dataset import Sample

def get_device(device_name='cuda'):

    if device_name == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def count_class_samples(samples, num_classes=2):

    counter = Counter((sample.label for sample in samples))
    return [max(counter.get(index, 0), 1) for index in range(num_classes)]

def move_batch_to_device(batch, device):

    images = batch['image'].to(device, non_blocking=True)
    labels = batch['label'].to(device, non_blocking=True)
    return (images, labels)
