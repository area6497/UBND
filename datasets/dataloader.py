
from pathlib import Path
from typing import Any
from torch.utils.data import DataLoader
from .breast_ultrasound_dataset import BreastUltrasoundDataset, Sample, load_samples_from_image_folder, load_samples_from_txt, split_k_folds

def _get_nested(config, keys, default=None):

    cursor = config
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor

def _dataset_kwargs(config, phase):

    data_cfg = config.get('data', {})
    preprocessing = data_cfg.get('preprocessing', {})
    normalize = preprocessing.get('normalize', {})
    nlm = preprocessing.get('nlm', {})
    augmentation = data_cfg.get('augmentation', {})
    return {'image_size': int(data_cfg.get('image_size', 224)), 'phase': phase, 'use_nlm_denoising': bool(preprocessing.get('use_nlm_denoising', True)), 'nlm_h': int(nlm.get('h', 10)), 'nlm_template_window_size': int(nlm.get('template_window_size', 7)), 'nlm_search_window_size': int(nlm.get('search_window_size', 21)), 'mean': tuple(normalize.get('mean', [0.485, 0.456, 0.406])), 'std': tuple(normalize.get('std', [0.229, 0.224, 0.225])), 'augment': bool(augmentation.get('enabled', True)), 'augment_factor': int(augmentation.get('expand_factor', 5))}

def _loader_kwargs(config, phase):

    train_cfg = config.get('train', {})
    test_cfg = config.get('test', {})
    is_train = phase == 'train'
    batch_size = train_cfg.get('batch_size') if is_train else test_cfg.get('batch_size')
    return {'batch_size': int(batch_size or 32), 'shuffle': is_train, 'num_workers': int(train_cfg.get('num_workers', 4)), 'pin_memory': bool(train_cfg.get('pin_memory', True)), 'drop_last': is_train}

def build_dataset(samples, config, phase):

    return BreastUltrasoundDataset(samples=samples, **_dataset_kwargs(config=config, phase=phase))

def build_loader(samples, config, phase):

    dataset = build_dataset(samples=samples, config=config, phase=phase)
    return DataLoader(dataset, **_loader_kwargs(config=config, phase=phase))

def load_samples_auto(data_dir=None, annotation_file=None, data_root=None):

    if annotation_file is not None and Path(annotation_file).exists():
        return load_samples_from_txt(annotation_file, data_root=data_root)
    if data_dir is not None:
        return load_samples_from_image_folder(data_dir)
    raise ValueError('The annotation_file or data_dir must be provided.')

def build_five_fold_loaders(samples, config):

    split_cfg = _get_nested(config, ['data', 'split'], {})
    num_folds = int(split_cfg.get('num_folds', 5))
    seed = int(_get_nested(config, ['project', 'seed'], 42))
    folds = split_k_folds(samples=samples, num_folds=num_folds, seed=seed)
    fold_loaders = []
    for train_samples, val_samples in folds:
        fold_loaders.append({'train': build_loader(train_samples, config=config, phase='train'), 'val': build_loader(val_samples, config=config, phase='val')})
    return fold_loaders

def build_public_five_fold_loaders(config):

    paths = config.get('paths', {})
    data_root = paths.get('data_root')
    dataset_specs = {'BUSI': {'data_dir': paths.get('busi_root'), 'annotation_file': paths.get('busi_list')}, 'UDIAT': {'data_dir': paths.get('udiat_root'), 'annotation_file': paths.get('udiat_list')}}
    loaders = {}
    for dataset_name, spec in dataset_specs.items():
        samples = load_samples_auto(data_dir=spec.get('data_dir'), annotation_file=spec.get('annotation_file'), data_root=data_root)
        loaders[dataset_name] = build_five_fold_loaders(samples=samples, config=config)
    return loaders

def build_train_val_test_loaders(config):

    paths = config.get('paths', {})
    data_root = paths.get('data_root')
    loaders = {}
    for phase, list_key in (('train', 'train_list'), ('val', 'val_list'), ('test', 'test_list')):
        annotation_file = paths.get(list_key)
        if annotation_file and Path(annotation_file).exists():
            samples = load_samples_from_txt(annotation_file, data_root=data_root)
            loaders[phase] = build_loader(samples=samples, config=config, phase=phase)
    if not loaders:
        raise ValueError('The train/val/test annotation file was not found.')
    return loaders

def build_external_loader(config):

    paths = config.get('paths', {})
    samples = load_samples_auto(data_dir=paths.get('private_root'), annotation_file=paths.get('private_list'), data_root=paths.get('data_root'))
    return build_loader(samples=samples, config=config, phase='external')
