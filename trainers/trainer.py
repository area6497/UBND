
from pathlib import Path
from typing import Any
import torch
from torch import nn
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
from tqdm import tqdm
from losses import TotalDistillationLoss
from metrics import compute_classification_metrics
from models import DenseNet121Teacher, UBNDStudent
from utils import AverageMeter, CSVLogger, move_batch_to_device, save_checkpoint, setup_logger

class UBNDTrainer:


    def __init__(self, config, train_loader, val_loader, class_counts, fold_name='single'):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.fold_name = fold_name
        self.device = torch.device('cuda' if torch.cuda.is_available() and config['train']['device'] == 'cuda' else 'cpu')
        self.output_dir = Path(config['paths']['logs_dir']) / fold_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(config['paths']['checkpoints_dir']) / fold_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger('UBNDTrainer', self.output_dir / 'train.log')
        self.csv_logger = CSVLogger(self.output_dir / 'history.csv', ['epoch', 'train_loss', 'val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1', 'val_auc'])
        self.writer = SummaryWriter(self.output_dir) if SummaryWriter is not None and config['train']['tensorboard']['enabled'] else None
        self.model = UBNDStudent(num_classes=config['project']['num_classes']).to(self.device)
        self.teacher = self._build_teacher()
        self.criterion = self._build_criterion(class_counts)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=config['train']['amp']['enabled'])
        self.best_acc = -1.0

    def fit(self):

        epochs = int(self.config['train']['epochs'])
        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss, metrics = self.validate(epoch)
            if self.scheduler is not None:
                self.scheduler.step()
            self.csv_logger.write({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': metrics.accuracy, 'val_precision': metrics.precision, 'val_recall': metrics.recall, 'val_f1': metrics.f1, 'val_auc': metrics.auc})
            self._write_tensorboard(epoch, train_loss, val_loss, metrics)
            self._save_epoch_checkpoint(epoch, metrics.accuracy)
        if self.writer is not None:
            self.writer.close()

    def train_one_epoch(self, epoch):

        self.model.train()
        if self.teacher is not None:
            self.teacher.eval()
        meter = AverageMeter()
        progress = tqdm(self.train_loader, desc=f'Train {self.fold_name} epoch {epoch}', leave=False)
        for batch in progress:
            images, labels = move_batch_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.config['train']['amp']['enabled']):
                student_logits = self.model(images)
                teacher_logits = None
                if self.teacher is not None:
                    with torch.no_grad():
                        teacher_logits = self.teacher(images)
                loss_dict = self.criterion(student_logits, labels, teacher_logits)
                loss = loss_dict['loss']
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            meter.update(loss.item(), n=images.size(0))
            progress.set_postfix(loss=f'{meter.avg:.4f}')
        return meter.avg

    @torch.no_grad()
    def validate(self, epoch):

        self.model.eval()
        meter = AverageMeter()
        all_logits = []
        all_labels = []
        progress = tqdm(self.val_loader, desc=f'Val {self.fold_name} epoch {epoch}', leave=False)
        for batch in progress:
            images, labels = move_batch_to_device(batch, self.device)
            logits = self.model(images)
            loss_dict = self.criterion(logits, labels, teacher_logits=None)
            meter.update(loss_dict['loss'].item(), n=images.size(0))
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
        logits_tensor = torch.cat(all_logits, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)
        metrics = compute_classification_metrics(labels_tensor, logits_tensor)
        self.logger.info('Epoch %03d | val_loss %.4f | Acc %.2f | Pr %.2f | Re %.2f | F1 %.2f | AUC %s', epoch, meter.avg, metrics.accuracy, metrics.precision, metrics.recall, metrics.f1, 'None' if metrics.auc is None else f'{metrics.auc:.4f}')
        return (meter.avg, metrics)

    def _build_teacher(self):

        teacher_cfg = self.config['teacher']
        if not teacher_cfg.get('enabled', True):
            return None
        teacher = DenseNet121Teacher(num_classes=teacher_cfg.get('num_classes', 2), pretrained=teacher_cfg.get('pretrained', True), checkpoint=teacher_cfg.get('weights'), freeze=teacher_cfg.get('freeze', True))
        return teacher.to(self.device)

    def _build_criterion(self, class_counts):

        loss_cfg = self.config['loss']
        lmf_cfg = loss_cfg['hybrid_lmf']
        return TotalDistillationLoss(class_counts=class_counts, alpha=loss_cfg.get('alpha', 0.5), temperature=loss_cfg.get('temperature', 6.0), focal_gamma=lmf_cfg['focal_loss'].get('gamma', 2.0), focal_alpha=lmf_cfg['focal_loss'].get('alpha'), ldam_max_m=lmf_cfg['ldam_loss'].get('max_m', 0.5), ldam_scale=lmf_cfg['ldam_loss'].get('scale', 30.0), lmf_weight=lmf_cfg.get('lmf_weight', 0.5))

    def _build_optimizer(self):

        opt_cfg = self.config['train']['optimizer']
        if opt_cfg['type'].lower() != 'adamw':
            raise ValueError('Use AdamW，implement only AdamW。')
        return torch.optim.AdamW(self.model.parameters(), lr=opt_cfg['lr'], weight_decay=opt_cfg['weight_decay'])

    def _build_scheduler(self):

        scheduler_cfg = self.config['train']['scheduler']
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config['train']['epochs'], eta_min=scheduler_cfg.get('min_lr', 1e-06))

    def _save_epoch_checkpoint(self, epoch, val_acc):
 
        save_checkpoint(self.checkpoint_dir / 'last.pth', model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, epoch=epoch, best_metric=max(self.best_acc, val_acc))
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            save_checkpoint(self.checkpoint_dir / 'best.pth', model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, epoch=epoch, best_metric=val_acc)

    def _write_tensorboard(self, epoch, train_loss, val_loss, metrics):

        if self.writer is None:
            return
        self.writer.add_scalar('loss/train', train_loss, epoch)
        self.writer.add_scalar('loss/val', val_loss, epoch)
        self.writer.add_scalar('metrics/accuracy', metrics.accuracy, epoch)
        self.writer.add_scalar('metrics/f1', metrics.f1, epoch)
        if metrics.auc is not None:
            self.writer.add_scalar('metrics/auc', metrics.auc, epoch)
