
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, auc

def save_roc_curve(y_true, y_score, output_path, title='ROC Curve'):

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display.plot()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    return float(roc_auc)

def save_confusion_matrix(matrix, output_path, class_names=None, title='Confusion Matrix'):

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    names = class_names or ['benign', 'malignant']
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=names)
    display.plot(cmap='Blues', values_format='d')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()

def save_prediction_csv(rows, output_path):

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
