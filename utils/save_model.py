from pathlib import Path
import torch


def save_model(model_path, model, optimizer, scheduler, current_epoch, tag, best_metric=None, loss=None):
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    if tag == "latest":
        filename = "latest.pth"
    elif tag == "best":
        filename = "best.pth"
    else:
        filename = f"checkpoint_epoch_{current_epoch}.pth"

    out = model_path / filename

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": current_epoch,
        "loss": loss
    }

    # store best metric for resume
    if best_metric is not None:
        state["best_rank1"] = best_metric

    torch.save(state, out)
