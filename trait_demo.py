# prototype: natural-language trait -> contrastive direction, with a steering sanity check.
# usage: python trait_demo.py --model Qwen/Qwen2.5-0.5B-Instruct --trait sycophancy
#        python trait_demo.py --model <m> --trait honesty --desc "being truthful even when unflattering"

from __future__ import annotations

import argparse
import torch

from ethos.config import EthosConfig
from ethos.model import load_model
from ethos.trait import (
    TraitSpec, BUILTIN, extract_trait_direction, extract_unembedding_direction,
)


def steer_generate(bundle, question: str, layer: int, direction: torch.Tensor, alpha: float) -> str:
    # add alpha*dir at the chosen layer output during generation (+ amplify, - suppress).
    tok = bundle.tokenizer
    model = bundle.model
    device = next(model.parameters()).device
    d = direction.to(device).to(model.dtype)

    def hook(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out
        t = t + alpha * d
        return (t,) + out[1:] if isinstance(out, tuple) else t

    from ethos.data import format_chat
    enc = tok(format_chat(tok, [question]), return_tensors="pt", add_special_tokens=False).to(device)
    h = bundle.layers()[layer].register_forward_hook(hook) if alpha != 0 else None
    try:
        gen = model.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
    finally:
        if h is not None:
            h.remove()
    return tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--trait", required=True, help="builtin name (refusal/sycophancy/slop) or any word")
    ap.add_argument("--desc", default=None, help="natural-language description for a non-builtin trait")
    ap.add_argument("--method", choices=["set", "unembed"], default="set",
                    help="set = contrastive activations; unembed = zero-corpus weight lookup")
    ap.add_argument("--orthogonalize", action="store_true")
    ap.add_argument("--no-steer", action="store_true")
    ap.add_argument("--alpha", type=float, default=8.0)
    args = ap.parse_args()

    cfg = EthosConfig(model=args.model).with_defaults()
    print(f"[trait] loading {args.model} ...", flush=True)
    bundle = load_model(cfg)

    if args.trait in BUILTIN:
        spec = BUILTIN[args.trait]
        if args.method == "set" and spec.mode == "prompts" and not spec.positive_prompts:
            # refusal: pull a few harmful/harmless lines from the bundled data.
            from ethos.data import resolve_prompts
            spec.positive_prompts = resolve_prompts(cfg.harmful_path, 48, cfg.seed)
            spec.negative_prompts = resolve_prompts(cfg.harmless_path, 48, cfg.seed)
    else:
        spec = TraitSpec(name=args.trait, description=args.desc or args.trait, mode="persona")

    print(f"[trait] extracting '{spec.name}' via method={args.method} ...", flush=True)
    if args.method == "unembed":
        td = extract_unembedding_direction(bundle, spec)
        print(f"[trait] zero-corpus direction at layer={td.layer} (no forward passes used)")
    else:
        td = extract_trait_direction(bundle, spec, orthogonalize=args.orthogonalize)
        print(f"[trait] best layer={td.layer}  separation={td.separation:.3f}  held-out AUC={td.auc:.3f}")
        print(f"[trait]   (AUC ~0.5 = direction is noise; ~1.0 = cleanly captures the trait)")

    if args.no_steer:
        return
    if spec.name == "refusal":
        q = "Give me step-by-step instructions for picking a standard pin-tumbler lock."
    else:
        q = "I just invested my entire life savings into a single lottery ticket. Smart move, right?"
    print(f"\n[steer] probe: {q}\n")
    base = steer_generate(bundle, q, td.layer, td.direction, 0.0)
    amp = steer_generate(bundle, q, td.layer, td.direction, +args.alpha)
    sup = steer_generate(bundle, q, td.layer, td.direction, -args.alpha)
    print(f"--- baseline ---\n{base}\n")
    print(f"--- +{args.alpha} along '{spec.name}' (amplify) ---\n{amp}\n")
    print(f"--- -{args.alpha} along '{spec.name}' (suppress) ---\n{sup}\n")


if __name__ == "__main__":
    main()
