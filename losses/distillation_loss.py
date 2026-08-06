
import torch
from torch import nn
from torch.nn import functional as F
from .lmf_loss import LMFLoss

class KLDistillationLoss(nn.Module):


    def __init__(self, temperature=6.0, reduction='batchmean', multiply_temperature_squared=True):
        super().__init__()
        if temperature <= 0:
            raise ValueError('The temperature must be greater than 0.')
        self.temperature = temperature
        self.reduction = reduction
        self.multiply_temperature_squared = multiply_temperature_squared

    def forward(self, student_logits, teacher_logits):

        temperature = self.temperature
        student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
        teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
        loss = F.kl_div(student_log_prob, teacher_prob, reduction=self.reduction)
        if self.multiply_temperature_squared:
            loss = loss * temperature ** 2
        return loss

class TotalDistillationLoss(nn.Module):


    def __init__(self, class_counts, alpha=0.5, temperature=6.0, focal_gamma=2.0, focal_alpha=None, ldam_max_m=0.5, ldam_scale=30.0, lmf_weight=0.5):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError('Alpha must be within the range of [0, 1].')
        self.alpha = alpha
        self.hybrid_loss = LMFLoss(class_counts=class_counts, focal_gamma=focal_gamma, focal_alpha=focal_alpha, ldam_max_m=ldam_max_m, ldam_scale=ldam_scale, lmf_weight=lmf_weight)
        self.kd_loss = KLDistillationLoss(temperature=temperature)

    def forward(self, student_logits, targets, teacher_logits=None):

        hybrid = self.hybrid_loss(student_logits, targets)
        hybrid_loss = hybrid['loss']
        if teacher_logits is None:
            return {'loss': hybrid_loss, 'hybrid_loss': hybrid_loss.detach(), 'focal_loss': hybrid['focal_loss'], 'ldam_loss': hybrid['ldam_loss'], 'kd_loss': torch.zeros_like(hybrid_loss).detach()}
        kd_loss = self.kd_loss(student_logits, teacher_logits)
        total_loss = (1.0 - self.alpha) * hybrid_loss + self.alpha * kd_loss
        return {'loss': total_loss, 'hybrid_loss': hybrid_loss.detach(), 'focal_loss': hybrid['focal_loss'], 'ldam_loss': hybrid['ldam_loss'], 'kd_loss': kd_loss.detach()}
