
import torch
from torch import nn
from torch.nn import functional as F

class FocalLoss(nn.Module):


    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is None:
            self.register_buffer('alpha', None)
        else:
            self.register_buffer('alpha', torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, targets):

        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1.0 - pt).pow(self.gamma)
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            focal_weight = focal_weight * alpha.gather(0, targets)
        loss = focal_weight * ce_loss
        return reduce_loss(loss, self.reduction)

class LDAMLoss(nn.Module):


    def __init__(self, class_counts, max_m=0.5, scale=30.0, reduction='mean'):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        if torch.any(counts <= 0):
            raise ValueError('The number of samples for each category in the "class_counts" must be greater than 0.')
        margins = 1.0 / torch.sqrt(torch.sqrt(counts))
        margins = margins * (max_m / margins.max())
        self.register_buffer('margins', margins)
        self.scale = scale
        self.reduction = reduction

    def forward(self, logits, targets):

        margins = self.margins.to(logits.device)
        batch_margins = margins.gather(0, targets)
        adjusted_logits = logits.clone()
        adjusted_logits[torch.arange(logits.size(0), device=logits.device), targets] -= batch_margins
        return F.cross_entropy(self.scale * adjusted_logits, targets, reduction=self.reduction)

class LMFLoss(nn.Module):


    def __init__(self, class_counts, focal_gamma=2.0, focal_alpha=None, ldam_max_m=0.5, ldam_scale=30.0, lmf_weight=0.5):
        super().__init__()
        if not 0.0 <= lmf_weight <= 1.0:
            raise ValueError('The lmf_weight must be within the range of [0, 1].')
        self.lmf_weight = lmf_weight
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.ldam = LDAMLoss(class_counts=class_counts, max_m=ldam_max_m, scale=ldam_scale)

    def forward(self, logits, targets):

        focal_loss = self.focal(logits, targets)
        ldam_loss = self.ldam(logits, targets)
        loss = self.lmf_weight * focal_loss + (1.0 - self.lmf_weight) * ldam_loss
        return {'loss': loss, 'focal_loss': focal_loss.detach(), 'ldam_loss': ldam_loss.detach()}

def reduce_loss(loss, reduction):

    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    if reduction == 'none':
        return loss
    raise ValueError(f'Unsupported reduction: {reduction}')
