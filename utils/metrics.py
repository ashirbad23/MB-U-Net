import torch


def confusion_matrix(preds, targets):
    preds = preds.view(-1)
    targets = targets.view(-1)

    TP = ((preds == 1) & (targets == 1)).sum().float()
    TN = ((preds == 0) & (targets == 0)).sum().float()
    FP = ((preds == 1) & (targets == 0)).sum().float()
    FN = ((preds == 0) & (targets == 1)).sum().float()

    return TP, TN, FP, FN


def compute_metrics_from_cm(TP, TN, FP, FN, eps=1e-6):
    iou = TP / (TP + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    mcc = (TP * TN - FP * FN) / torch.sqrt(
        (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + eps
    )

    return {
        "IoU": iou.item(),
        "Precision": precision.item(),
        "Recall": recall.item(),
        "F1": f1.item(),
        "MCC": mcc.item()
    }


def compute_metrics(preds, targets, threshold=0.5):
    """
    preds: logits [B,1,H,W]
    targets: [B,1,H,W]
    """
    probs = torch.sigmoid(preds)
    preds_bin = (probs > threshold).float()

    TP, TN, FP, FN = confusion_matrix(preds_bin, targets)
    return compute_metrics_from_cm(TP, TN, FP, FN)


def find_best_threshold(preds, targets, thresholds=None):
    """
    preds: logits [B,1,H,W]
    targets: [B,1,H,W]
    """
    if thresholds is None:
        thresholds = torch.linspace(0.2, 0.8, 25)

    probs = torch.sigmoid(preds)

    best_mcc = -1
    best_t = 0.5
    best_metrics = None

    for t in thresholds:
        preds_bin = (probs > t).float()
        TP, TN, FP, FN = confusion_matrix(preds_bin, targets)
        metrics = compute_metrics_from_cm(TP, TN, FP, FN)

        if metrics["MCC"] > best_mcc:
            best_mcc = metrics["MCC"]
            best_t = t.item()
            best_metrics = metrics

    return best_t, best_metrics
