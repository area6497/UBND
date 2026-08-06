
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

class ClassificationMetricResult:

    def __init__(self, accuracy, precision, recall, sensitivity, specificity, f1, auc, confusion_matrix):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.f1 = f1
        self.auc = auc
        self.confusion_matrix = confusion_matrix


    def to_dict(self):

        return {'accuracy': self.accuracy, 'precision': self.precision, 'recall': self.recall, 'sensitivity': self.sensitivity, 'specificity': self.specificity, 'f1': self.f1, 'auc': self.auc, 'confusion_matrix': self.confusion_matrix.astype(int).tolist()}

def compute_classification_metrics(y_true, y_score, threshold=0.5, positive_label=1):

    targets = to_numpy(y_true).astype(int)
    scores = to_numpy(y_score)
    positive_scores = get_positive_scores(scores, positive_label=positive_label)
    predictions = (positive_scores >= threshold).astype(int)
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    accuracy = accuracy_score(targets, predictions) * 100.0
    precision = precision_score(targets, predictions, zero_division=0) * 100.0
    recall = recall_score(targets, predictions, zero_division=0) * 100.0
    specificity = safe_divide(tn, tn + fp) * 100.0
    f1 = f1_score(targets, predictions, zero_division=0) * 100.0
    auc = compute_auc(targets, positive_scores)
    return ClassificationMetricResult(accuracy=float(accuracy), precision=float(precision), recall=float(recall), sensitivity=float(recall), specificity=float(specificity), f1=float(f1), auc=auc, confusion_matrix=matrix)

def get_positive_scores(scores, positive_label=1):

    if scores.ndim == 1:
        return scores.astype(float)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError('The y_score must be of the form [N] or [N, C].')
    row_sums = scores.sum(axis=1)
    is_probability = np.all(scores >= 0) and np.all(scores <= 1) and np.allclose(row_sums, 1, atol=0.0001)
    probabilities = scores if is_probability else softmax_numpy(scores)
    return probabilities[:, positive_label]

def compute_auc(targets, positive_scores):

    if len(np.unique(targets)) < 2:
        return None
    return float(roc_auc_score(targets, positive_scores))

def to_numpy(values):

    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)

def softmax_numpy(logits):

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)

def safe_divide(numerator, denominator):

    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)
