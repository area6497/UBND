
from pathlib import Path
from typing import Callable, Iterable
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
DEFAULT_CLASS_TO_INDEX = {'benign': 0, 'malignant': 1}

class Sample:

    def __init__(self, image_path, label):
        self.image_path = image_path
        self.label = label
    'Information of a single image sample. \n\n    Attributes:\n        image_path: Absolute or relative path of the image file。\n        label: Category Index，benign=0，malignant=1。\n    '

class BreastUltrasoundDataset(Dataset):


    def __init__(self, samples, image_size=224, phase='train', use_nlm_denoising=True, nlm_h=10, nlm_template_window_size=7, nlm_search_window_size=21, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), augment=True, augment_factor=5):
        self.samples = list(samples)
        if not self.samples:
            raise ValueError('The samples cannot be empty. Please check the data path or the annotation file.')
        self.image_size = image_size
        self.phase = phase
        self.use_nlm_denoising = use_nlm_denoising
        self.nlm_h = nlm_h
        self.nlm_template_window_size = nlm_template_window_size
        self.nlm_search_window_size = nlm_search_window_size
        self.is_train = phase == 'train'
        self.augment_factor = max(1, int(augment_factor)) if self.is_train else 1
        self.transform = self._build_transform(image_size=image_size, mean=mean, std=std, augment=augment and self.is_train)

    def __len__(self):

        return len(self.samples) * self.augment_factor

    def __getitem__(self, index):
    
        sample = self.samples[index % len(self.samples)]
        image = self._read_image(sample.image_path)
        if self.use_nlm_denoising:
            image = self._apply_nlm_denoising(image)
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(sample.label, dtype=torch.long)
        return {'image': image_tensor, 'label': label_tensor, 'path': str(sample.image_path)}

    @staticmethod
    def _build_transform(image_size, mean, std, augment):
        
        if augment:
            return transforms.Compose([transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(degrees=15), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02), transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5), transforms.RandomPerspective(distortion_scale=0.2, p=0.3), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        return transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    @staticmethod
    def _read_image(image_path):
        
        if not image_path.exists():
            raise FileNotFoundError(f'The image does not exist: {image_path}')
        return Image.open(image_path).convert('RGB')

    def _apply_nlm_denoising(self, image):
     
        image_array = np.array(image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        denoised_bgr = cv2.fastNlMeansDenoisingColored(image_bgr, None, h=self.nlm_h, hColor=self.nlm_h, templateWindowSize=self.nlm_template_window_size, searchWindowSize=self.nlm_search_window_size)
        denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(denoised_rgb)

def load_samples_from_txt(annotation_file, data_root=None):

    annotation_path = Path(annotation_file)
    if not annotation_path.exists():
        raise FileNotFoundError(f'The annotation file does not exist: {annotation_path}')
    root = Path(data_root) if data_root is not None else annotation_path.parent
    samples = []
    for line_number, line in enumerate(annotation_path.read_text(encoding='utf-8').splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            image_text, label_text = stripped.rsplit(maxsplit=1)
        except ValueError as exc:
            raise ValueError(f'The format of line {line_number} in the annotation file is incorrect. It should be: image_path label') from exc
        image_path = Path(image_text)
        if not image_path.is_absolute():
            image_path = root / image_path
        samples.append(Sample(image_path=image_path, label=int(label_text)))
    return samples

def load_samples_from_image_folder(data_dir, class_to_index=None):

    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f'Data directory does not exist: {root}')
    mapping = class_to_index or DEFAULT_CLASS_TO_INDEX
    samples = []
    for class_name, label in mapping.items():
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.rglob('*')):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append(Sample(image_path=image_path, label=label))
    if not samples:
        raise ValueError(f'No benign/malignant images were found in {root}.')
    return samples

def split_k_folds(samples, num_folds=5, seed=42):

    sample_list = list(samples)
    if num_folds < 2:
        raise ValueError('num_folds must >= 2。')
    rng = np.random.default_rng(seed)
    by_class = {}
    for sample in sample_list:
        by_class.setdefault(sample.label, []).append(sample)
    class_folds = {}
    for label, class_samples in by_class.items():
        shuffled = class_samples.copy()
        rng.shuffle(shuffled)
        class_folds[label] = [list(chunk) for chunk in np.array_split(shuffled, num_folds)]
    folds = []
    for fold_index in range(num_folds):
        val_samples = []
        train_samples = []
        for chunks in class_folds.values():
            val_samples.extend(chunks[fold_index])
            for index, chunk in enumerate(chunks):
                if index != fold_index:
                    train_samples.extend(chunk)
        folds.append((train_samples, val_samples))
    return folds
