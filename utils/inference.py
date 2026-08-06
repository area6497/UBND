
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from models import UBNDStudent
from utils import apply_overrides, load_checkpoint, load_config, resolve_paths
PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def parse_args():

    parser = argparse.ArgumentParser(description='Inference with UBND_final')
    parser.add_argument('--config', default=str(PROJECT_ROOT / 'configs' / 'config.yaml'))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True, help='single image or folder')
    parser.add_argument('--output', default=None, help='Output CSV path')
    parser.add_argument('--override', action='append', default=None)
    return parser.parse_args()

def main():

    args = parse_args()
    config = resolve_paths(apply_overrides(load_config(args.config), args.override), PROJECT_ROOT)
    device = torch.device('cuda' if torch.cuda.is_available() and config['train']['device'] == 'cuda' else 'cpu')
    model = UBNDStudent(num_classes=config['project']['num_classes']).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device, strict=True)
    model.eval()
    image_paths = collect_images(Path(args.input))
    transform = build_inference_transform(config)
    rows = []
    with torch.no_grad():
        for image_path in tqdm(image_paths, desc='Inference'):
            image = read_rgb_image(image_path, config)
            tensor = transform(image).unsqueeze(0).to(device)
            prob = model.predict_proba(tensor)[0].cpu().numpy()
            pred = int(np.argmax(prob))
            rows.append({'path': str(image_path), 'pred_label': pred, 'pred_name': config['project']['class_names'][pred], 'prob_benign': float(prob[0]), 'prob_malignant': float(prob[1])})
    output_path = Path(args.output) if args.output else Path(config['paths']['outputs_dir']) / 'predictions' / 'inference.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved inference results to {output_path}')

def collect_images(input_path):

    if input_path.is_file():
        return [input_path]
    images = [path for path in sorted(input_path.rglob('*')) if path.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        raise FileNotFoundError(f'Image not found: {input_path}')
    return images

def build_inference_transform(config):

    data_cfg = config['data']
    normalize = data_cfg['preprocessing']['normalize']
    return transforms.Compose([transforms.Resize((data_cfg['image_size'], data_cfg['image_size'])), transforms.ToTensor(), transforms.Normalize(mean=normalize['mean'], std=normalize['std'])])

def read_rgb_image(image_path, config):

    image = Image.open(image_path).convert('RGB')
    preprocessing = config['data']['preprocessing']
    if not preprocessing.get('use_nlm_denoising', True):
        return image
    nlm = preprocessing['nlm']
    image_array = np.array(image)
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(image_bgr, None, h=nlm.get('h', 10), hColor=nlm.get('h', 10), templateWindowSize=nlm.get('template_window_size', 7), searchWindowSize=nlm.get('search_window_size', 21))
    return Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
if __name__ == '__main__':
    main()
