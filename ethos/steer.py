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


def clamp_hooks(bundle: ModelBundle, direction: torch.Tensor, layers: List[int], target: float) -> List:
    # set the residual's component along `direction` to a fixed target (projection clamp), instead
    # of adding a fixed vector. removes the prompt-dependent baseline -> consistent across prompts.
    device = next(bundle.model.parameters()).device
    d = (direction / direction.norm()).to(device).to(bundle.model.dtype)
    handles = []

    def mk(_l):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            coord = (t @ d).unsqueeze(-1)
            t = t - coord * d + target * d
            return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
        return hook

    for l in layers:
        handles.append(bundle.layers()[l].register_forward_hook(mk(l)))
    return handles


def band_clamp_hooks(bundle: ModelBundle, plan: dict, amp: float, gen_only: bool = False) -> List:
    # the production steering: per-layer trait clamp across a band (consistent + bounded), plus a set
    # of pinned axes held at fixed values -- language axis to neutral (no off-language drift) and the
    # disclaimer/hedge axis low (stay opinionated). amp scales the trait target neutral -> in-trait.
    model = bundle.model
    device = next(model.parameters()).device
    dt = model.dtype
    pins = [(p["dir"].to(device).to(dt), p["targets"]) for p in plan.get("pins", [])
            if float(p["dir"].norm()) > 1e-6]
    handles = []

    def mk(d, tgt, layer):
        layer_pins = [(pd, pt[layer]) for pd, pt in pins]

        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            t = t.clone()
            sl = t[:, -1, :] if gen_only else t
            sl = sl - (sl @ d).unsqueeze(-1) * d + tgt * d
            for pd, pv in layer_pins:
                sl = sl - (sl @ pd).unsqueeze(-1) * pd + pv * pd
            if gen_only:
                t[:, -1, :] = sl
            else:
                t = sl
            return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
        return hook

    # keep the target in-distribution: hi is a real in-trait anchor (allow some headroom past it),
    # but lo is only neutral -- there's no anti-trait anchor, so negative amp is blind extrapolation
    # and must stay close. beyond this the clamp forces an unseen coordinate and the model derails.
    amp = max(-1.0, min(2.0, amp))
    for l in plan["band"]:
        d = plan["dirs"][l].to(device).to(dt)
        tgt = plan["lo"][l] + amp * (plan["hi"][l] - plan["lo"][l])
        handles.append(bundle.layers()[l].register_forward_hook(mk(d, tgt, l)))
    return handles


def refusal_ablation_hooks(bundle: ModelBundle) -> List:
    # project the refusal direction out of every layer during steered generation, so the model
    # won't refuse or deflect a task the steered persona should just do (e.g. a rude char that won't
    # roast, a blunt char that hedges). the trait is useless if safety training overrides it -- this
    # is the same apostate-style ablation used during elicitation, now applied at inference too.
    from .trait import _refusal_directions, _refusal_ablation_hooks
    return _refusal_ablation_hooks(bundle, _refusal_directions(bundle))


def cjk_logits_processor(bundle: ModelBundle):
    # hard guarantee against language drift: ban cjk tokens in the logits during steered generation.
    # the activation pin keeps coherence; this stops any chinese token from being emitted at all.
    from transformers import LogitsProcessor, LogitsProcessorList
    ids = getattr(bundle, "_cjk_ids", None)
    if not ids:
        return None
    banned = torch.tensor(ids, device=next(bundle.model.parameters()).device)

    class _Ban(LogitsProcessor):
        def __call__(self, input_ids, scores):
            scores[:, banned] = float("-inf")
            return scores

    return LogitsProcessorList([_Ban()])
