from pathlib import Path
import torch


def save_model(model_path, model, optimizer, scheduler, current_epoch, loss):
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    out = model_path / f"checkpointV1_{current_epoch}_{loss:.4f}.pth"
    state = {'net': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(),
             'epoch': current_epoch}
    torch.save(state, out)
