import torch


def _confusion_matrix(preds, targets):
    """
    preds, targets: [B, 1, H, W] (binary 0/1)
    """
    preds = preds.view(-1)
    targets = targets.view(-1)

    TP = ((preds == 1) & (targets == 1)).sum().float()
    TN = ((preds == 0) & (targets == 0)).sum().float()
    FP = ((preds == 1) & (targets == 0)).sum().float()
    FN = ((preds == 0) & (targets == 1)).sum().float()

    return TP, TN, FP, FN


def iou_score(preds, targets, eps=1e-6):
    TP, _, FP, FN = _confusion_matrix(preds, targets)
    return TP / (TP + FP + FN + eps)


def precision_score(preds, targets, eps=1e-6):
    TP, _, FP, _ = _confusion_matrix(preds, targets)
    return TP / (TP + FP + eps)


def recall_score(preds, targets, eps=1e-6):
    TP, _, _, FN = _confusion_matrix(preds, targets)
    return TP / (TP + FN + eps)


def f1_score(preds, targets, eps=1e-6):
    precision = precision_score(preds, targets, eps)
    recall = recall_score(preds, targets, eps)
    return 2 * (precision * recall) / (precision + recall + eps)


def mcc_score(preds, targets, eps=1e-6):
    TP, TN, FP, FN = _confusion_matrix(preds, targets)

    numerator = (TP * TN) - (FP * FN)
    denominator = torch.sqrt(
        (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + eps
    )

    return numerator / (denominator + eps)


def compute_metrics(preds, targets):
    """
    preds: logits [B, 1, H, W]
    targets: [B, 1, H, W]
    """
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    return {
        "IoU": iou_score(preds, targets).item(),
        "Precision": precision_score(preds, targets).item(),
        "Recall": recall_score(preds, targets).item(),
        "F1": f1_score(preds, targets).item(),
        "MCC": mcc_score(preds, targets).item()
    }