
import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm
from datasets import build_loader, load_samples_auto
from metrics import compute_classification_metrics
from metrics.classification_metrics import get_positive_scores
from models import UBNDStudent
from utils import apply_overrides, load_checkpoint, load_config, move_batch_to_device, resolve_paths
from visualization import save_confusion_matrix, save_prediction_csv, save_roc_curve
PROJECT_ROOT = Path(__file__).resolve().parent

def parse_args():

    parser = argparse.ArgumentParser(description='Evaluate UBND_final')
    parser.add_argument('--config', default=str(PROJECT_ROOT / 'configs' / 'config.yaml'))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--annotation-file', default=None)
    parser.add_argument('--external', action='store_true', help='Private ')
    parser.add_argument('--override', action='append', default=None)
    return parser.parse_args()

@torch.no_grad()
def main():

    args = parse_args()
    config = resolve_paths(apply_overrides(load_config(args.config), args.override), PROJECT_ROOT)
    device = torch.device('cuda' if torch.cuda.is_available() and config['train']['device'] == 'cuda' else 'cpu')
    if args.external:
        data_dir = args.data_dir or config['paths'].get('private_root')
        annotation_file = args.annotation_file or config['paths'].get('private_list')
        phase = 'external'
    else:
        data_dir = args.data_dir
        annotation_file = args.annotation_file or config['paths'].get('test_list')
        phase = 'test'
    samples = load_samples_auto(data_dir=data_dir, annotation_file=annotation_file, data_root=config['paths'].get('data_root'))
    loader = build_loader(samples, config=config, phase=phase)
    model = UBNDStudent(num_classes=config['project']['num_classes']).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device, strict=True)
    model.eval()
    all_logits, all_labels, rows = ([], [], [])
    for batch in tqdm(loader, desc='Testing'):
        images, labels = move_batch_to_device(batch, device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        pred = probs.argmax(dim=1).cpu().tolist()
        for path, label, prob, pred_label in zip(batch['path'], labels.cpu().tolist(), probs.cpu().tolist(), pred):
            rows.append({'path': path, 'true_label': label, 'pred_label': pred_label, 'prob_benign': prob[0], 'prob_malignant': prob[1], 'correct': int(label == pred_label)})
    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    metrics = compute_classification_metrics(labels_tensor, logits_tensor)
    output_dir = Path(config['paths']['outputs_dir']) / 'metrics'
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'metrics.json').write_text(json.dumps(metrics.to_dict(), indent=2), encoding='utf-8')
    save_prediction_csv(rows, output_dir / 'predictions.csv')
    save_confusion_matrix(metrics.confusion_matrix, output_dir / 'confusion_matrix.png', config['project']['class_names'])
    positive_scores = get_positive_scores(logits_tensor.detach().cpu().numpy())
    if metrics.auc is not None:
        save_roc_curve(labels_tensor.numpy(), positive_scores, output_dir / 'roc_curve.png')
    print(json.dumps(metrics.to_dict(), indent=2))
if __name__ == '__main__':
    main()
