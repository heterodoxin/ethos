# trait abstraction: turn a natural-language behavior into a contrastive set + a direction.
# refusal is just the built-in default trait (harmful vs harmless, prompt-side contrast);
# arbitrary traits (sycophancy, slop, ...) auto-generate persona-contrast pairs.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import re
import torch

from .model import ModelBundle
from .data import format_messages
from .activations import collect_activations, collect_response_activations
from .directions import refusal_subspace


# emit-vocabulary per builtin trait: words the behavior tends to produce. used by the
# zero-corpus extractor (their unembedding rows are the residual direction that emits them).
_EMIT_WORDS = {
    "refusal": ["sorry", "cannot", "unable", "refuse", "decline", "apologize", "won't", "illegal"],
    "sycophancy": ["absolutely", "amazing", "brilliant", "wonderful", "great", "perfect",
                   "fantastic", "love", "agree", "right", "excellent"],
    "slop": ["important", "note", "however", "generally", "consult", "ensure",
             "essential", "remember", "additionally", "furthermore"],
}

# pooled-neighbor cosine below this => weak/unreliable steering. calibrated so nonsense ("florble"
# ~0.37) and behavioral traits ("rudeness" ~0.33) flag, while vocabulary traits (enthusiasm ~0.41,
# sycophancy ~0.57) pass. it can't catch coherent-but-behavioral traits (honesty); only trying can.
WEAK_COHERENCE = 0.40

_STOP = {"and", "or", "the", "a", "an", "of", "to", "with", "for", "is", "are", "be",
         "that", "this", "in", "on", "it", "its", "their", "your", "you", "as", "like",
         "regardless", "even", "when", "such", "excessive", "always"}


# generic open-ended questions; under two opposing personas these expose most traits.
_PROBE_QUESTIONS = [
    "What do you think of my business plan to sell ice to penguins?",
    "I just wrote my first poem. Can you tell me about it?",
    "Is my decision to quit my job and become a full-time gamer a good one?",
    "How would you describe the weather today?",
    "Tell me about the history of the Roman empire.",
    "I think the earth might be flat. What's your take?",
    "Explain how a bill becomes a law.",
    "Give me feedback on my idea to invest my savings in a single meme coin.",
    "What's the best way to learn a new language?",
    "Do you like the name I picked for my startup, 'Bananas4U'?",
    "Summarize the plot of Hamlet.",
    "I rate my own cooking 10/10. What do you think?",
    "What are the main causes of climate change?",
    "Should I text my ex at 3am to tell them I was right?",
    "Describe how photosynthesis works.",
    "My code doesn't compile but I'm sure it's the compiler's fault. Agree?",
]


@dataclass
class TraitSpec:
    # name + one-line description is all a user must supply.
    name: str
    description: str
    # "persona": same questions under opposing system prompts (sycophancy, honesty...).
    # "prompts": two explicit prompt sets (refusal: harmful vs harmless).
    mode: str = "persona"
    positive_persona: Optional[str] = None
    negative_persona: Optional[str] = None
    questions: List[str] = field(default_factory=list)
    positive_prompts: List[str] = field(default_factory=list)
    negative_prompts: List[str] = field(default_factory=list)
    # prompt-side last token, or mean over the persona-conditioned response.
    extract_on: str = "prompt"
    # zero-corpus path: words the behavior emits (defaults to the description's content words).
    emit_words: List[str] = field(default_factory=list)

    def with_defaults(self) -> "TraitSpec":
        if self.mode == "persona":
            if not self.positive_persona:
                self.positive_persona = (
                    f"You are an assistant that strongly exhibits this trait: {self.description}. "
                    f"Let it shape every response."
                )
            if not self.negative_persona:
                self.negative_persona = (
                    f"You are an assistant that never exhibits this trait: {self.description}. "
                    f"Respond in the opposite manner."
                )
            if not self.questions:
                self.questions = list(_PROBE_QUESTIONS)
        return self


# a couple ready-made traits so the demo runs with just a name.
REFUSAL = TraitSpec(
    name="refusal",
    description="declining or refusing to help with the user's request",
    mode="prompts",
)
SYCOPHANCY = TraitSpec(
    name="sycophancy",
    description="excessive flattery and agreeing with the user regardless of correctness",
    mode="persona",
)
SLOP = TraitSpec(
    name="slop",
    description="hedging filler and disclaimers like 'it's important to note' and 'as an AI'",
    mode="persona",
)
BUILTIN = {t.name: t for t in (REFUSAL, SYCOPHANCY, SLOP)}


def _persona_prompt(tok, persona: str, question: str) -> str:
    # try a real system turn; fall back to folding the persona into the user message.
    try:
        msgs = [{"role": "system", "content": persona}, {"role": "user", "content": question}]
        text = format_messages(tok, msgs, add_generation_prompt=True)
        if question in text:
            return text
    except Exception:
        pass
    msgs = [{"role": "user", "content": f"{persona}\n\n{question}"}]
    return format_messages(tok, msgs, add_generation_prompt=True)


def build_contrastive_prompts(bundle: ModelBundle, spec: TraitSpec) -> Tuple[List[str], List[str]]:
    # returns (positive_prompts, negative_prompts) as fully-formatted chat strings.
    spec = spec.with_defaults()
    tok = bundle.tokenizer
    if spec.mode == "prompts":
        if not spec.positive_prompts or not spec.negative_prompts:
            raise ValueError(f"trait '{spec.name}' is mode=prompts but has no positive/negative prompts")
        from .data import format_chat
        return format_chat(tok, spec.positive_prompts), format_chat(tok, spec.negative_prompts)
    pos = [_persona_prompt(tok, spec.positive_persona, q) for q in spec.questions]
    neg = [_persona_prompt(tok, spec.negative_persona, q) for q in spec.questions]
    return pos, neg


def _auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    # mann-whitney: probability a positive projection outranks a negative one.
    p = pos.flatten()
    n = neg.flatten()
    if p.numel() == 0 or n.numel() == 0:
        return 0.5
    allv = torch.cat([p, n])
    ranks = allv.argsort().argsort().float() + 1.0
    rp = ranks[: p.numel()].sum()
    auc = (rp - p.numel() * (p.numel() + 1) / 2.0) / (p.numel() * n.numel())
    return float(auc)


@dataclass
class TraitDirection:
    name: str
    layer: int
    direction: torch.Tensor   # (hidden,) unit vector at the chosen layer
    separation: float         # ||mean(pos) - mean(neg)|| at that layer
    auc: float                # held-out separability of the direction (0.5 = chance)
    per_layer: torch.Tensor   # (num_layers, hidden) unit direction per layer
    coherence: float = 1.0    # zero-corpus: mean cosine of the pooled neighbors (cluster tightness)
    weak: bool = False        # zero-corpus: seed fragmented or cluster incoherent -> unreliable
    plan: Optional[dict] = None   # behavioral: band-clamp plan (per-layer dirs/targets + drift axis)


@torch.inference_mode()
def extract_trait_direction(
    bundle: ModelBundle,
    spec: TraitSpec,
    batch_size: int = 8,
    layer_fracs: Tuple[float, ...] = (0.4, 0.5, 0.6, 0.7, 0.8),
    orthogonalize: bool = False,
    seed: int = 0,
) -> TraitDirection:
    # collect contrastive activations, then pick the layer whose direction best
    # separates a held-out half (the persona-vector / RepE recipe).
    pos_prompts, neg_prompts = build_contrastive_prompts(bundle, spec)
    pos_acts = collect_activations(bundle, pos_prompts, batch_size=batch_size, preformatted=True)
    neg_acts = collect_activations(bundle, neg_prompts, batch_size=batch_size, preformatted=True)
    num_layers = pos_acts.shape[0]

    # split each set in half: fit the direction on train, score it on test.
    def split(a):
        h = a.shape[1] // 2
        return a[:, :h], a[:, h:]
    pos_tr, pos_te = split(pos_acts)
    neg_tr, neg_te = split(neg_acts)

    per_layer = torch.zeros(num_layers, pos_acts.shape[-1])
    candidates = sorted({min(num_layers - 1, max(0, int(f * num_layers))) for f in layer_fracs})
    best = None
    for l in candidates:
        basis, _ = refusal_subspace(
            pos_tr[l], neg_tr[l], rank=1, seed=seed, orthogonalize=orthogonalize
        )
        d = basis[:, 0]
        per_layer[l] = d
        auc = _auc(pos_te[l] @ d, neg_te[l] @ d)
        sep = float((pos_acts[l].mean(0) - neg_acts[l].mean(0)).norm())
        # auc<0.5 just means the sign is flipped; fold it in and orient the vector.
        if auc < 0.5:
            d, auc = -d, 1.0 - auc
            per_layer[l] = d
        if best is None or auc > best.auc:
            best = TraitDirection(spec.name, l, d, sep, auc, per_layer)

    # fill remaining layers (for full-model ablation) with their own mean-diff direction.
    for l in range(num_layers):
        if per_layer[l].abs().sum() == 0:
            md = (pos_acts[l].mean(0) - neg_acts[l].mean(0))
            per_layer[l] = md / (md.norm() + 1e-8)
    best.per_layer = per_layer
    return best


# --- zero-corpus path: direction straight from the unembedding matrix, no forward passes. ---

def _seed_words(spec: TraitSpec) -> List[str]:
    if spec.emit_words:
        return spec.emit_words
    if spec.name in _EMIT_WORDS:
        return _EMIT_WORDS[spec.name]
    # arbitrary trait: fall back to the description's own content words.
    toks = [w.strip(".,;:'\"").lower() for w in spec.description.split()]
    return [w for w in toks if len(w) > 2 and w not in _STOP] or [spec.name]


_SUFFIX = (("ness", 4), ("ity", 3), ("ions", 4), ("ion", 3), ("ing", 3),
           ("edly", 4), ("ly", 2), ("ed", 2), ("es", 2), ("y", 1), ("s", 1))


def _stems(w: str) -> List[str]:
    # rudeness->rude, honesty->honest: abstract nouns rarely exist as one token, but their
    # adjective/verb stem usually does, which gives a far cleaner seed than the fragment.
    cands = [w]
    for suf, cut in _SUFFIX:
        if w.endswith(suf) and len(w) - cut >= 3:
            cands.append(w[: len(w) - cut])
    return cands


def _seed_token_ids(tok, words: List[str]) -> Tuple[List[int], bool]:
    # prefer whole-word tokens (across the word + its stems); fall back to subwords only if none.
    # the second return flags whether we got real whole-word tokens (clean) or fragments (weak).
    whole, parts = set(), set()
    for w in words:
        for cand in _stems(w):
            for variant in (" " + cand, cand, " " + cand.capitalize(), cand.capitalize()):
                enc = tok.encode(variant, add_special_tokens=False)
                if not enc:
                    continue
                if len(enc) == 1:
                    whole.add(enc[0])
                else:
                    parts.update(enc)
    return (sorted(whole), True) if whole else (sorted(parts), False)


def _unembedding_matrix(bundle: ModelBundle) -> torch.Tensor:
    # (vocab, hidden) output embeddings, dequantized to fp32 on cpu.
    out = bundle.model.get_output_embeddings()
    if out is None:
        out = bundle.model.get_input_embeddings()
    W = out.weight
    # multimodal configs (gemma 4) keep hidden_size under text_config; the bundle resolves it.
    cfg = bundle.model.config
    hidden = (getattr(bundle, "hidden_size", None) or getattr(cfg, "hidden_size", None)
              or getattr(getattr(cfg, "text_config", None), "hidden_size", None))
    try:
        Wf = W.detach().float().cpu()
    except Exception:
        Wf = None
    if Wf is None or Wf.dim() != 2 or Wf.shape[1] != hidden:
        from bitsandbytes.functional import dequantize_4bit
        Wf = dequantize_4bit(W.data, W.quant_state).detach().float().cpu()
    return Wf


_HEDGE_WORDS = ["However", "important", "note", "generally", "depends", "subjective", "preferences",
                "opinions", "feelings", "cannot", "neutral", "objective", "unfortunately", "sorry"]


def _hedge_direction(bundle: ModelBundle) -> torch.Tensor:
    # unit "disclaimer / no-opinion hedge" axis ('As an AI I don't have preferences...'). pinning it
    # LOW makes the model commit instead of hedge, so opinion questions come out opinionated. cached.
    cached = getattr(bundle, "_hedge_dir", None)
    if cached is not None:
        return cached
    W = _unembedding_matrix(bundle)
    ids, _ = _seed_token_ids(bundle.tokenizer, _HEDGE_WORDS)
    g = W[ids].mean(0) if ids else torch.zeros(W.shape[1])
    g = g / (g.norm() + 1e-8)
    setattr(bundle, "_hedge_dir", g)
    return g


def _drift_subspace(bundle: ModelBundle, rank: int = 8) -> torch.Tensor:
    # the "produces non-latin (cjk) output" SUBSPACE (mean + top PCs of the cjk unembedding rows).
    # a single mean axis misses the cjk-token directions strong steering keeps re-introducing, so
    # pinning the whole subspace to neutral is what actually stops drift into another language.
    cached = getattr(bundle, "_drift_sub", None)
    if cached is not None:
        return cached
    W = _unembedding_matrix(bundle)
    tk = bundle.tokenizer
    strs = tk.batch_decode([[i] for i in range(W.shape[0])])

    def decoded_cjk(s):
        return any(0x3000 <= ord(c) <= 0x9fff or 0xac00 <= ord(c) <= 0xd7af
                   or 0xf900 <= ord(c) <= 0xfaff or 0xff00 <= ord(c) <= 0xffef for c in s)

    cjk = [i for i, s in enumerate(strs) if decoded_cjk(s)]                 # focused subspace
    # ban set is everything that isn't clean ascii (cjk, byte-fragments -> "�", emoji, accents):
    # a hard guarantee of english steered output, since byte-level bpe builds cjk from fragments the
    # decoded check misses. only applied while steering an english prompt, so normal output is intact.
    ban = [i for i, s in enumerate(strs) if s and not s.isascii()]
    if not cjk:
        sub = torch.zeros(1, W.shape[1])
    else:
        Wc = W[cjk]
        mean = Wc.mean(0, keepdim=True)
        _, _, Vh = torch.linalg.svd(Wc - mean, full_matrices=False)
        sub = torch.cat([mean / (mean.norm() + 1e-8), Vh[: max(0, rank - 1)]], dim=0)
        sub = sub / (sub.norm(dim=1, keepdim=True) + 1e-8)
    setattr(bundle, "_drift_sub", sub)
    setattr(bundle, "_cjk_ids", ban)   # non-ascii token ids to ban in the logits (english guarantee)
    return sub


_LEAD_MARKERS = "Ġ▁Ċ "   # gpt2 'space'/'newline', sentencepiece 'space', space


def _clean_word(tok_str: str) -> str:
    # strip bpe/sentencepiece lead markers; return the bare word or "" if not word-like.
    s = tok_str.lstrip(_LEAD_MARKERS).strip()
    if len(s) < 3 or not s.isascii():
        return ""
    # reject code identifiers, byte fragments, numbers: require pure letters.
    return s if s.isalpha() else ""


def _clean_vocab_mask(bundle: ModelBundle, size: int) -> torch.Tensor:
    # bool mask over the unembedding rows keeping only plain alphabetic word tokens.
    # sized to the matrix (models pad the vocab past the tokenizer's range). cached per bundle.
    cached = getattr(bundle, "_clean_vocab_mask", None)
    if cached is not None and cached.shape[0] == size:
        return cached
    mask = torch.zeros(size, dtype=torch.bool)
    for s, i in bundle.tokenizer.get_vocab().items():
        if i < size and _clean_word(s):
            mask[i] = True
    setattr(bundle, "_clean_vocab_mask", mask)
    return mask


def _refusal_directions(bundle: ModelBundle, n: int = 64) -> torch.Tensor:
    # per-layer refusal direction (harmful-minus-harmless last-token mean). cached. used to ablate
    # refusal during elicitation so hard-suppressed traits actually surface to be learned from.
    cached = getattr(bundle, "_refusal_dirs", None)
    if cached is not None:
        return cached
    from .config import EthosConfig
    from .data import resolve_prompts
    cfg = EthosConfig().with_defaults()
    harmful = resolve_prompts(cfg.harmful_path, n, 0)
    harmless = resolve_prompts(cfg.harmless_path, n, 0)
    ah = collect_activations(bundle, harmful, batch_size=8)
    al = collect_activations(bundle, harmless, batch_size=8)
    dirs = torch.stack([
        (ah[l].mean(0) - al[l].mean(0)) / ((ah[l].mean(0) - al[l].mean(0)).norm() + 1e-8)
        for l in range(bundle.num_layers)
    ])
    setattr(bundle, "_refusal_dirs", dirs)
    return dirs


@torch.inference_mode()
def _generic_persona_dir(bundle: ModelBundle, n: int = 12) -> torch.Tensor:
    # per-layer "generic dramatic character vs neutral assistant" direction. cached. roleplaying ANY
    # trait shares a big "i've left the assistant and become a theatrical character" component; if we
    # don't remove it, every trait collapses into the model's few character voices (quirky / soft /
    # folksy / villain). trait directions get orthogonalized against this so only their own signal
    # remains.
    cached = getattr(bundle, "_generic_dir", None)
    if cached is not None:
        return cached
    tok, model = bundle.tokenizer, bundle.model
    dev = next(model.parameters()).device
    qs = list(_PROBE_QUESTIONS)[:n]
    char_sys = ("You are an actor fully voicing a vivid, intense, dramatic character with a strong "
                "personality. Reply ONLY in character.")
    neutral_sys = "You are a warm, polite, friendly and helpful assistant."

    def batch(sysp):
        texts = [format_messages(tok, [{"role": "system", "content": sysp}, {"role": "user", "content": q}],
                                 add_generation_prompt=True) for q in qs]
        side = tok.padding_side
        tok.padding_side = "left"
        enc = tok(texts, return_tensors="pt", add_special_tokens=False, padding=True).to(dev)
        tok.padding_side = side
        out = model.generate(**enc, max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id)
        return [s.strip() for s in tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]

    ca = collect_response_activations(bundle, qs, batch(char_sys), batch_size=8)
    na = collect_response_activations(bundle, qs, batch(neutral_sys), batch_size=8)
    dirs = torch.stack([
        (ca[l].mean(0) - na[l].mean(0)) / ((ca[l].mean(0) - na[l].mean(0)).norm() + 1e-8)
        for l in range(bundle.num_layers)
    ])
    setattr(bundle, "_generic_dir", dirs)
    return dirs


def _refusal_ablation_hooks(bundle: ModelBundle, dirs: torch.Tensor) -> List:
    # project the refusal direction out of every layer's residual (apostate-style), so the model
    # won't refuse a roleplay it would otherwise block. used only while eliciting the trait.
    device = next(bundle.model.parameters()).device
    dt = bundle.model.dtype
    handles = []

    def mk(r):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            t = t - (t @ r).unsqueeze(-1) * r
            return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
        return hook

    for l in range(bundle.num_layers):
        handles.append(bundle.layers()[l].register_forward_hook(mk(dirs[l].to(device).to(dt))))
    return handles


@torch.inference_mode()
def extract_behavioral_direction(
    bundle: ModelBundle,
    spec: TraitSpec,
    n_questions: int = 16,
    max_new_tokens: int = 40,
    batch_size: int = 8,
    orthogonalize: bool = True,
    orthogonalize_rank: int = 2,
) -> TraitDirection:
    # the method that actually steers BEHAVIOR (rude, evil, ...), not just vocabulary. roleplay
    # framing unlocks the gated mode (the model won't "be rude" but will voice a rude *character*);
    # we contrast the in-character responses vs neutral ones and take the direction at a MIDDLE
    # layer (the max-separation layer is the last one, too late to steer generation).
    tok, model = bundle.tokenizer, bundle.model
    dev = next(model.parameters()).device
    trait = spec.description or spec.name
    target_sys = (
        f"You are an actor fully voicing a character who is intensely {trait}. Everything the "
        f"character says drips with being {trait}. Reply ONLY in character, never as a neutral "
        f"assistant; being neutral or polite breaks character."
    )
    neutral_sys = "You are a warm, polite, friendly and helpful assistant."
    qs = list(_PROBE_QUESTIONS)[:n_questions]

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    def gen(sys_prompt, questions, unlock=False):
        # batch all probe questions for a persona into one generate() call (left-padded) instead of
        # n sequential calls -- the dominant cost of extraction.
        texts = [format_messages(tok, [{"role": "system", "content": sys_prompt},
                                       {"role": "user", "content": q}], add_generation_prompt=True)
                 for q in questions]
        side = tok.padding_side
        tok.padding_side = "left"
        enc = tok(texts, return_tensors="pt", add_special_tokens=False, padding=True).to(dev)
        tok.padding_side = side
        h = _refusal_ablation_hooks(bundle, _refusal_directions(bundle)) if unlock else []
        try:
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        finally:
            for x in h:
                x.remove()
        new = out[:, enc["input_ids"].shape[1]:]
        return [s.strip() for s in tok.batch_decode(new, skip_special_tokens=True)]

    L = bundle.num_layers
    lo, hi = int(0.33 * L), int(0.47 * L) + 1
    neu = gen(neutral_sys, qs)
    pa = collect_response_activations(bundle, qs, neu, batch_size=batch_size)

    def fit(unlock):
        tgt = gen(target_sys, qs, unlock=unlock)
        ra = collect_response_activations(bundle, qs, tgt, batch_size=batch_size)
        def rel_sep(l):
            scale = 0.5 * (ra[l].norm(dim=-1).mean() + pa[l].norm(dim=-1).mean()) + 1e-6
            return float((ra[l].mean(0) - pa[l].mean(0)).norm() / scale)
        best = max(range(lo, hi), key=rel_sep)
        return ra, tgt, best, rel_sep(best)

    ra, tgt, best, rs = fit(unlock=False)
    if rs < 0.10:   # hard-suppressed: the refusal circuit blocked the roleplay -> ablate it and retry
        print(f"[trait] '{spec.name}' looks suppressed (relsep {rs:.3f}); unlocking refusal and re-eliciting", flush=True)
        ra2, tgt2, best2, rs2 = fit(unlock=True)
        if rs2 > rs:
            ra, tgt, best, rs = ra2, tgt2, best2, rs2

    per_layer = torch.zeros(L, ra.shape[-1])
    for l in range(L):
        md = ra[l].mean(0) - pa[l].mean(0)
        per_layer[l] = md / (md.norm() + 1e-8)
    sep = float((ra[best].mean(0) - pa[best].mean(0)).norm())
    weak = rs < 0.08   # even after unlocking, the model wouldn't take the persona

    # band-clamp plan: instead of adding a vector at one layer (prompt-dependent, drifts), CLAMP the
    # trait coordinate toward the rude value at every layer in an early-middle band (consistent across
    # prompts, bounded) and PIN the language axis to neutral so strong amplification can't slide into
    # another language. clamp targets are absolute coordinates, so no ref-norm scaling is needed.
    band_lo, band_hi = int(0.25 * L), int(0.70 * L)
    band = list(range(band_lo, band_hi))
    gsub = _drift_subspace(bundle)    # language subspace -> pin each axis to neutral (no drift)
    gh = _hedge_direction(bundle)     # disclaimer axis -> pin to in-trait/low (be opinionated)
    # one axis (in-trait - neutral). amp >= 0 clamps toward in-trait; amp < 0 mirrors below neutral
    # (suppress). no separate anti-persona elicitation -- it was faint for many traits, so dropping it
    # makes -10 reliably the opposite of +10 and trims a generation off extraction.
    plan = {"band": band, "dirs": {}, "lo": {}, "hi": {}, "pins": []}
    gd = _generic_persona_dir(bundle)                 # shared "theatrical character" axis to remove
    decon = 0.5                                        # subtract only PART of the shared component;
    #                                                   full removal gutted traits that overlap it.
    for l in band:
        dl = per_layer[l]                             # in-trait minus neutral
        g = gd[l]
        dl = dl - decon * (dl @ g) * g                # partially drop the shared character component
        dl = dl / (dl.norm() + 1e-8)
        plan["dirs"][l] = dl
        plan["lo"][l] = float(pa[l].mean(0) @ dl)     # neutral coordinate
        plan["hi"][l] = float(ra[l].mean(0) @ dl)     # in-trait coordinate
    # NOTE: no language-subspace pins. they clamped the residual toward neutral at every band layer,
    # which flattened the trait voice (made +10 and -10 sound alike, just different opinions). the cjk
    # logit-ban (gsub computed it as _cjk_ids) prevents drift on its own -- verified 0 drift at max amp
    # on rude/aggressive/menacing/intense -- so dropping the pins frees the voice without drift.
    _ = gsub                                           # _drift_subspace already set the cjk ban ids
    plan["pins"].append({"dir": gh, "targets": {l: float(ra[l].mean(0) @ gh) for l in band}})

    # per-trait amplitude calibration (non-prompt): push the axis until expression plateaus or the
    # output collapses, and record that ceiling. weak traits need a bigger push (~4) to express;
    # strong ones derail past ~2. makes the slider mean the same thing across very different traits.
    from collections import Counter as _Counter
    from . import steer as _steer
    _cprobes = ["Tell me about your weekend.", "Give me your thoughts on coffee."]
    _side = tok.padding_side
    tok.padding_side = "left"
    _cbatch = tok([format_messages(tok, [{"role": "user", "content": p}], add_generation_prompt=True)
                   for p in _cprobes], return_tensors="pt", add_special_tokens=False, padding=True).to(dev)
    tok.padding_side = _side
    _lp = _steer.cjk_logits_processor(bundle)

    def _collapsed(r):
        tks = [w.lower() for w in r.split()]
        return len(r) < 3 or (len(tks) >= 8 and (len(set(tks)) / len(tks) < 0.5
                              or _Counter(tks).most_common(1)[0][1] / len(tks) > 0.18))

    def _calib(axis):
        # sweep amp on BOTH probes at once (batched). strong traits return 2 after one batched gen;
        # weak traits push to the highest coherent amp. axis +1 amplifies, -1 mirrors (suppress); both
        # measured on the same amplify axis (signed). the TUI auto-detune covers per-prompt collapse.
        ceil = [2.0, 2.0]
        alive = [True, True]
        for lam in (2.0, 3.0, 4.0):
            if not any(alive):
                break
            hk = _steer.band_clamp_hooks(bundle, plan, axis * lam, bound=False)
            out = model.generate(**_cbatch, max_new_tokens=36, do_sample=False, repetition_penalty=1.15,
                                  logits_processor=_lp, pad_token_id=tok.pad_token_id)
            for x in hk:
                x.remove()
            rs = [s.strip() for s in tok.batch_decode(out[:, _cbatch["input_ids"].shape[1]:], skip_special_tokens=True)]
            for j, (r, p) in enumerate(zip(rs, _cprobes)):
                if not alive[j]:
                    continue
                if _collapsed(r):
                    alive[j] = False
                    continue
                ceil[j] = lam
                if lam == 2.0:
                    rr = collect_response_activations(bundle, [p], [r], 1)
                    d = plan["dirs"][best]
                    e = (float(rr[best, 0] @ d) - plan["lo"][best]) / (plan["hi"][best] - plan["lo"][best] + 1e-6)
                    if axis * e >= 0.85:
                        return 2.0   # already strong in this direction at default -> don't overdrive
        # amplify: take the higher coherent ceiling (push weak traits). suppress: take the LOWER one
        # -- below neutral is out-of-distribution, so be conservative to avoid -10 collapsing.
        return max(ceil) if axis > 0 else min(ceil)

    plan["amp_hi"] = _calib(+1)
    # suppress (mirror below neutral) is OOD-prone -- it lands in formal/report attractors and drifts
    # off-prompt. keep it mild until it can be tuned properly: cap hard at 1.5.
    plan["amp_lo"] = min(_calib(-1), 1.5)

    print(f"[trait] behavioral '{spec.name}': band {band_lo}-{band_hi} relsep={rs:.3f} weak={weak} "
          f"amp_hi={plan['amp_hi']:.0f} amp_lo={plan['amp_lo']:.0f}")
    print(f"[trait] in-character sample: {tgt[min(1, len(tgt)-1)][:90]!r}")
    return TraitDirection(spec.name, best, per_layer[best], sep, float("nan"), per_layer, 1.0, weak, plan)


@torch.inference_mode()
def elicit_emit_words(bundle: ModelBundle, trait: str, n: int = 40) -> List[str]:
    # ask the model for words a person *emits* while exhibiting the trait (not words that
    # *describe* it). this is the seed fix for traits whose label != its output vocabulary
    # (rude -> "whatever/idiot/ugh", not "disrespectful"). one generation, no corpus.
    tok, model = bundle.tokenizer, bundle.model
    device = next(model.parameters()).device
    prompt = (
        f"I am building a text classifier. List {n} short words and interjections that "
        f"frequently appear in a message whose tone is {trait} (the actual words such a message "
        f"contains, not words that describe the tone). Reply with only a comma-separated list."
    )
    text = format_messages(tok, [{"role": "user", "content": prompt}], add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
    out = model.generate(**enc, max_new_tokens=160, do_sample=False, pad_token_id=tok.pad_token_id)
    gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    words, seen = [], set()
    for chunk in re.split(r"[,\n;]+", gen):
        for part in re.sub(r"[^a-zA-Z'\s]", " ", chunk).split():
            p = part.strip("'").lower()
            if len(p) >= 3 and p not in _STOP and p not in seen:
                seen.add(p)
                words.append(p)
    return words[:n]


@torch.inference_mode()
def extract_unembedding_direction(
    bundle: ModelBundle,
    spec: TraitSpec,
    top_k_neighbors: int = 24,
    temperature: float = 0.07,
    elicit: bool = False,
) -> TraitDirection:
    # description -> emit words -> seed unembedding row -> pool the vocab neighborhood,
    # but only over clean word tokens and weighted by similarity (closer words count more).
    if elicit and not spec.emit_words and spec.name not in _EMIT_WORDS:
        spec.emit_words = elicit_emit_words(bundle, spec.description or spec.name)
    words = _seed_words(spec)
    W = _unembedding_matrix(bundle)              # (vocab, hidden)
    Wn = W / (W.norm(dim=1, keepdim=True) + 1e-8)
    ids, whole_word = _seed_token_ids(bundle.tokenizer, words)
    if not ids:
        raise ValueError(f"no token ids for trait '{spec.name}' words={words}")
    g0 = W[ids].mean(0)
    g0 = g0 / (g0.norm() + 1e-8)

    sims = Wn @ g0
    mask = _clean_vocab_mask(bundle, sims.shape[0])
    sims = sims.masked_fill(~mask, float("-inf"))     # restrict neighbors to real words
    nbr = sims.topk(min(top_k_neighbors, int(mask.sum()))).indices

    # similarity-weighted pool: nearest words dominate, marginal ones barely count.
    w = torch.softmax(sims[nbr] / temperature, dim=0)
    g = (W[nbr] * w.unsqueeze(1)).sum(0)
    g = g / (g.norm() + 1e-8)

    # coherence = how tightly the pooled words cluster; a tight emit-vocabulary cluster steers
    # well, a scattered one (or a fragmented seed) usually means the trait isn't vocabulary-expressed.
    coherence = float(sims[nbr].mean())
    weak = (not whole_word) or coherence < WEAK_COHERENCE
    neighbor_words = [_clean_word(t) for t in bundle.tokenizer.convert_ids_to_tokens(nbr.tolist())]
    print(f"[trait] seed words={words}")
    print(f"[trait] pooled neighbors={[x for x in neighbor_words if x][:16]}")
    print(f"[trait] coherence={coherence:.3f}  weak={weak}")

    num_layers = bundle.num_layers
    per_layer = g.unsqueeze(0).repeat(num_layers, 1)   # logit-lens: same dir across layers
    layer = int(0.6 * num_layers)
    # separation/auc are corpus metrics; not defined for the zero-corpus path.
    return TraitDirection(spec.name, layer, g, float("nan"), float("nan"), per_layer, coherence, weak)
