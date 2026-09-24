from __future__ import annotations

import copy
import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        src = dict(model.named_parameters())
        for name, p_ema in self.shadow.named_parameters():
            p = src[name]
            p_ema.lerp_(p.detach(), 1.0 - self.decay)
        src_b = dict(model.named_buffers())
        for name, b_ema in self.shadow.named_buffers():
            if name in src_b:
                b_ema.copy_(src_b[name])

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state):
        self.shadow.load_state_dict(state)
