from pathlib import Path
import torch


def save_model(model_path, model, optimizer, scheduler, current_epoch, tag, **kwargs):
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    if tag == "latest":
        filename = "latest.pth"
    elif tag == "best":
        filename = "best.pth"
    else:
        filename = f"checkpoint_epoch_{current_epoch}.pth"

    out = model_path / filename

    # 🔥 base state
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": current_epoch,
    }

    # 🔥 dynamically add extra things
    for key, value in kwargs.items():
        if hasattr(value, "state_dict"):
            state[key] = value.state_dict()
        else:
            state[key] = value

    torch.save(state, out)