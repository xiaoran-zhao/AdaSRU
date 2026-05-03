from __future__ import annotations

from pathlib import Path

import torch


def extract_named_adam_diag(model, optimizer) -> dict[str, torch.Tensor]:
    diag = {}
    for name, param in model.named_parameters():
        state = optimizer.state.get(param, {})
        if 'exp_avg_sq' in state:
            diag[name] = state['exp_avg_sq'].detach().cpu().clone()
    return diag


def save_training_checkpoint(path: str | Path, model, optimizer, args: dict, metric: dict) -> None:
    path = Path(path)
    payload = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'adam_diag_by_name': extract_named_adam_diag(model, optimizer),
        'args': args,
        'metric': metric,
    }
    torch.save(payload, path)
