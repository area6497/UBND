
import argparse
from pathlib import Path
from datasets import build_loader, load_samples_auto, split_k_folds
from trainers import UBNDTrainer
from utils import apply_overrides, count_class_samples, load_config, resolve_paths, save_config, set_seed
PROJECT_ROOT = Path(__file__).resolve().parent

def parse_args():

    parser = argparse.ArgumentParser(description='Train UBND_final')
    parser.add_argument('--config', default=str(PROJECT_ROOT / 'configs' / 'config.yaml'))
    parser.add_argument('--mode', choices=['single', 'five-fold'], default='single')
    parser.add_argument('--dataset', choices=['BUSI', 'UDIAT'], default='BUSI')
    parser.add_argument('--data-dir', default=None, help='ImageFolder data directory')
    parser.add_argument('--annotation-file', default=None, help='txt annotation file')
    parser.add_argument('--val-file', default=None, help='single Pattern validation set txt')
    parser.add_argument('--override', action='append', default=None, help='Coverage configuration，such as train.batch_size=16')
    return parser.parse_args()

def main():

    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args.override)
    config = resolve_paths(config, PROJECT_ROOT)
    set_seed(config['project']['seed'], config['project'].get('deterministic', False))
    if args.mode == 'five-fold':
        dataset_key = args.dataset.lower()
        samples = load_samples_auto(data_dir=args.data_dir or config['paths'].get(f'{dataset_key}_root'), annotation_file=args.annotation_file or config['paths'].get(f'{dataset_key}_list'), data_root=config['paths'].get('data_root'))
        folds = split_k_folds(samples, num_folds=config['data']['split']['num_folds'], seed=config['project']['seed'])
        for fold_index, (train_samples, val_samples) in enumerate(folds, start=1):
            fold_name = f'{args.dataset}_fold{fold_index}'
            train_loader = build_loader(train_samples, config=config, phase='train')
            val_loader = build_loader(val_samples, config=config, phase='val')
            trainer = UBNDTrainer(config=config, train_loader=train_loader, val_loader=val_loader, class_counts=count_class_samples(train_samples, config['project']['num_classes']), fold_name=fold_name)
            save_config(config, Path(config['paths']['logs_dir']) / fold_name / 'config.yaml')
            trainer.fit()
        return
    train_samples = load_samples_auto(data_dir=args.data_dir, annotation_file=args.annotation_file or config['paths'].get('train_list'), data_root=config['paths'].get('data_root'))
    val_samples = load_samples_auto(annotation_file=args.val_file or config['paths'].get('val_list'), data_root=config['paths'].get('data_root'))
    train_loader = build_loader(train_samples, config=config, phase='train')
    val_loader = build_loader(val_samples, config=config, phase='val')
    trainer = UBNDTrainer(config=config, train_loader=train_loader, val_loader=val_loader, class_counts=count_class_samples(train_samples, config['project']['num_classes']), fold_name='single')
    save_config(config, Path(config['paths']['logs_dir']) / 'single' / 'config.yaml')
    trainer.fit()
if __name__ == '__main__':
    main()
