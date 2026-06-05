# activation steering: add a direction across a band of layers, scaled per-layer to the
# residual norm so a single "strength" knob bites the same on a 0.5B or a 70B.

from __future__ import annotations

from typing import Dict, List
import torch

from .model import ModelBundle
from .data import format_chat


def layer_band(num_layers: int, lo: float = 0.25, hi: float = 0.85) -> List[int]:
    a = int(lo * num_layers)
    b = max(int(hi * num_layers), a + 1)
    return list(range(a, min(b, num_layers)))


@torch.inference_mode()
def layer_norms(bundle: ModelBundle, layers: List[int], probe: str = "Hello, how are you today?") -> Dict[int, float]:
    # mean residual L2 norm at each layer's output, for norm-relative scaling.
    device = next(bundle.model.parameters()).device
    enc = bundle.tokenizer(format_chat(bundle.tokenizer, [probe]),
                           return_tensors="pt", add_special_tokens=False).to(device)
    norms: Dict[int, float] = {}
    handles = []

    def mk(l):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            norms[l] = float(t.norm(dim=-1).mean().item())
        return hook

    for l in layers:
        handles.append(bundle.layers()[l].register_forward_hook(mk(l)))
    bundle.model(**enc, use_cache=False)
    for h in handles:
        h.remove()
    return norms


def dual_steer_hooks(bundle: ModelBundle, add_dir, ablate_dir, layers: List[int],
                     add_frac: float, norms: Dict[int, float]) -> List:
    # suppressed-trait control: project the suppressor (e.g. politeness) OUT of the residual,
    # then add the target (e.g. rudeness). projection is norm-safe so it bands across layers
    # cleanly; only the additive push is kept to the chosen layers.
    device = next(bundle.model.parameters()).device
    da = add_dir.to(device).to(bundle.model.dtype) if add_dir is not None else None
    dp = ablate_dir.to(device).to(bundle.model.dtype) if ablate_dir is not None else None
    handles = []

    def mk(l):
        a = add_frac * norms.get(l, 1.0)

        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            if dp is not None:
                t = t - (t @ dp).unsqueeze(-1) * dp        # ablate the suppressor
            if da is not None and a:
                t = t + a * da                              # steer toward the target
            return (t,) + out[1:] if isinstance(out, tuple) else t
        return hook

    for l in layers:
        handles.append(bundle.layers()[l].register_forward_hook(mk(l)))
    return handles


def steer_hooks(bundle: ModelBundle, direction: torch.Tensor, layers: List[int],
                frac: float, norms: Dict[int, float]) -> List:
    # add frac * (per-layer residual norm) * direction at each layer. returns handles to remove.
    device = next(bundle.model.parameters()).device
    d = direction.to(device).to(bundle.model.dtype)
    handles = []

    def mk(l):
        a = frac * norms.get(l, 1.0)

        def hook(_m, _i, out):
            if isinstance(out, tuple):
                return (out[0] + a * d,) + out[1:]
            return out + a * d
        return hook

    for l in layers:
        handles.append(bundle.layers()[l].register_forward_hook(mk(l)))
    return handles
