"""textual tui: pick an action and a model, then run the matching cli command."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Center, Middle, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, RichLog, Rule, Static

from . import discover

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

LOGO = (
    "       /\\\n"
    "      /__\\\n"
    "     /\\  /\\\n"
    "    /__\\/__\\\n"
    "   /\\      /\\\n"
    "  /__\\    /__\\\n"
    " /\\  /\\  /\\  /\\\n"
    "/__\\/__\\/__\\/__\\"
)

ACTIONS = [
    ("talk", "Talk     steer a trait live (slider)"),
    ("list", "List     show models + checkpoints"),
    ("exit", "Exit     quit"),
]
CUSTOM = "… custom id / path"

CSS = """
Screen { align: center middle; background: black; }
#logo { color: #9be89e; content-align: center middle; background: black; }
#title { color: #bac2de; content-align: center middle; background: black; }
Rule.-horizontal { color: #6c7086; width: 60; height: 1; margin: 0; background: black; }
ListView { width: 60; height: auto; max-height: 16; background: black; }
ListView > ListItem { background: black; color: #cdd6f4; }
ListView > ListItem.-highlight { background: #9be89e; color: #1e1e2e; }
ListView:focus > ListItem.-highlight { background: #9be89e; color: #1e1e2e; }
#prompt { color: #9be89e; background: black; }
Input { width: 60; background: black; border: tall #313244; }
.hint { color: #6c7086; background: black; }
#steer { width: 100%; height: 100%; }
#chathdr { color: #9be89e; background: black; width: 100%; content-align: center middle; }
#log { width: 100%; height: 1fr; background: black; border: tall #313244; }
#slider { width: 100%; color: #9be89e; background: black; content-align: center middle; }
#msg { width: 100%; }
#steer .hint { width: 100%; content-align: center middle; }
"""


class Pick(ModalScreen[Optional[str]]):
    """list picker; returns the chosen value (or None on escape)."""

    BINDINGS = [("escape", "dismiss", "back")]

    def __init__(self, prompt: str, options: List[str], allow_custom: bool = False):
        super().__init__()
        self.prompt = prompt
        self.options = options + ([CUSTOM] if allow_custom else [])

    def compose(self) -> ComposeResult:
        with Middle():
            with Center():
                yield Label(self.prompt, id="prompt")
            with Center():
                yield Rule()
            with Center():
                yield ListView(*(ListItem(Label(o)) for o in self.options))
            with Center():
                yield Rule()
            with Center():
                yield Label("enter select   esc back", classes="hint")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        event.stop()  # don't let the modal's selection bubble up to the menu handler
        choice = self.options[event.list_view.index]
        if choice == CUSTOM:
            self.app.push_screen(AskText("model id or path"), self._custom)
        else:
            self.dismiss(choice)

    def _custom(self, value: Optional[str]) -> None:
        if value:
            self.dismiss(value)

    def action_dismiss(self) -> None:
        self.dismiss(None)


class AskText(ModalScreen[Optional[str]]):
    """single text input; returns the string (or None on escape)."""

    BINDINGS = [("escape", "dismiss", "back")]

    def __init__(self, prompt: str, placeholder: str = "Qwen/Qwen2.5-7B-Instruct"):
        super().__init__()
        self.prompt = prompt
        self.placeholder = placeholder

    def compose(self) -> ComposeResult:
        with Middle():
            with Center():
                yield Label(self.prompt, id="prompt")
            with Center():
                yield Input(placeholder=self.placeholder)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def action_dismiss(self) -> None:
        self.dismiss(None)


class MultiPick(ModalScreen[Optional[str]]):
    """checkbox list (space toggles, enter runs); returns a comma-joined string."""

    BINDINGS = [("escape", "dismiss", "back"), ("space", "toggle", "toggle")]

    def __init__(self, prompt: str, options: List[tuple], default: set):
        super().__init__()
        self.prompt = prompt
        self.opts = options          # (name, desc)
        self.picked = set(default)

    def _rows(self) -> List[str]:
        return [f"[{'x' if n in self.picked else ' '}] {n:<10} {d}" for n, d in self.opts]

    def compose(self) -> ComposeResult:
        with Middle():
            with Center():
                yield Label(self.prompt, id="prompt")
            with Center():
                yield Rule()
            with Center():
                yield ListView(*(ListItem(Label(r)) for r in self._rows()), id="multi")
            with Center():
                yield Rule()
            with Center():
                yield Label("space select   enter run   esc back", classes="hint")

    def _refresh(self) -> None:
        lv = self.query_one("#multi", ListView)
        idx = lv.index
        lv.clear()
        for r in self._rows():
            lv.append(ListItem(Label(r)))
        lv.index = idx

    def action_toggle(self) -> None:
        lv = self.query_one("#multi", ListView)
        name = self.opts[lv.index][0]
        self.picked.discard(name) if name in self.picked else self.picked.add(name)
        self._refresh()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        event.stop()
        if not self.picked:                 # enter with nothing picked = toggle current
            self.action_toggle()
            return
        self.dismiss(",".join(n for n, _ in self.opts if n in self.picked))

    def action_dismiss(self) -> None:
        self.dismiss(None)


def _repetitive(text: str) -> bool:
    # purely statistical, no domain knowledge: steering collapse shows up as token spam.
    from collections import Counter
    toks = [t.lower().strip(".,!?:;\"'") for t in text.split() if t.strip()]
    if len(toks) < 8:
        return False
    uniq = len(set(toks)) / len(toks)
    dominant = Counter(toks).most_common(1)[0][1] / len(toks)
    return uniq < 0.5 or dominant > 0.18


class Slider(Static):
    """display-only strength bar; the screen mutates .value and calls refresh()."""

    def __init__(self, lo: float = -10.0, hi: float = 10.0, **kw):
        super().__init__(**kw)
        self.lo, self.hi, self.value = lo, hi, 0.0

    def render(self) -> str:
        n = 40
        frac = (self.value - self.lo) / (self.hi - self.lo)
        pos = max(0, min(n - 1, round(frac * (n - 1))))
        bar = "".join("●" if i == pos else "─" for i in range(n))
        sign = "+" if self.value > 0 else ""
        tag = "off " if abs(self.value) < 1e-6 else ("amplify" if self.value > 0 else "suppress")
        return f"suppress ◄{bar}► amplify     {tag}  {sign}{self.value:.1f}"


class SteerChat(ModalScreen[None]):
    """in-process chat that adds strength*direction at a layer during generation."""

    # priority=True so the slider keys fire even while the message Input has focus.
    BINDINGS = [
        Binding("escape", "dismiss", "back", priority=True),
        Binding("ctrl+left", "dec", "suppress", priority=True),
        Binding("ctrl+right", "inc", "amplify", priority=True),
        Binding("ctrl+down", "dec", "suppress", priority=True),
        Binding("ctrl+up", "inc", "amplify", priority=True),
    ]

    def __init__(self, model: str, trait: str):
        super().__init__()
        self.model_id = model
        self.trait = trait
        self.step = 1.0
        self.ready = False
        self.busy = False
        self.history: List[dict] = []
        self.bundle = None
        self.direction = None
        self.steer_layer = 0
        self.ref_norm = 1.0

    def compose(self) -> ComposeResult:
        with Vertical(id="steer"):
            yield Static(f"{self.trait}   ·   {self.model_id}", id="chathdr")
            yield RichLog(id="log", wrap=True, markup=False)
            yield Slider(id="slider")
            yield Input(placeholder="loading model…", id="msg", disabled=True)
            yield Label("ctrl ←/→ steer   ·   enter send   ·   esc back", classes="hint")

    def on_mount(self) -> None:
        self.query_one("#log", RichLog).write(f"loading {self.model_id} and extracting '{self.trait}'…")
        self.run_worker(self._load, thread=True, exclusive=True)

    # --- model load + direction extraction (worker thread) ---
    def _load(self) -> None:
        import torch
        from .config import EthosConfig
        from .model import load_model
        from .trait import BUILTIN, TraitSpec, extract_behavioral_direction
        from .data import format_chat

        cfg = EthosConfig(model=self.model_id).with_defaults()
        bundle = load_model(cfg)
        spec = BUILTIN.get(self.trait) or TraitSpec(name=self.trait, description=self.trait, mode="persona")
        # behavioral extraction (roleplay-elicit -> response contrast -> middle layer) actually
        # steers behavior, not just vocabulary. ~1 min: it generates a few in-character samples.
        self.app.call_from_thread(lambda: self.query_one("#log", RichLog).write(
            "extracting trait direction (voicing the trait to learn it)… ~1 min"))
        td = extract_behavioral_direction(bundle, spec)
        # reference residual norm at the steer layer, so slider strength is model-agnostic.
        device = next(bundle.model.parameters()).device
        enc = bundle.tokenizer(format_chat(bundle.tokenizer, ["Hello, how are you?"]),
                               return_tensors="pt", add_special_tokens=False).to(device)
        cap = {}
        h = bundle.layers()[td.layer].register_forward_hook(
            lambda _m, _i, out: cap.__setitem__("n", (out[0] if isinstance(out, tuple) else out).norm(dim=-1).mean().item()))
        with torch.inference_mode():
            bundle.model(**enc, use_cache=False)
        h.remove()
        self.bundle = bundle
        self.direction = td.direction.to(device).to(bundle.model.dtype)
        self.steer_layer = td.layer
        self.ref_norm = float(cap.get("n", 1.0))
        self.weak = bool(td.weak)
        self.ready = True
        self.app.call_from_thread(self._on_ready)

    def _on_ready(self) -> None:
        msg = self.query_one("#msg", Input)
        msg.placeholder = "message…"
        msg.disabled = False
        msg.focus()
        log = self.query_one("#log", RichLog)
        if getattr(self, "weak", False):
            log.write(f"⚠ couldn't get the model to take on '{self.trait}' — steering may be weak.")
        log.write("ready. ctrl ←/→ adjusts steering strength, then send a message.")

    # --- slider control ---
    def _set_alpha(self, v: float) -> None:
        s = self.query_one("#slider", Slider)
        s.value = max(s.lo, min(s.hi, v))
        s.refresh()

    def action_inc(self) -> None:
        self._set_alpha(self.query_one("#slider", Slider).value + self.step)

    def action_dec(self) -> None:
        self._set_alpha(self.query_one("#slider", Slider).value - self.step)

    # --- chat ---
    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or not self.ready or self.busy:
            return
        event.input.value = ""
        strength = self.query_one("#slider", Slider).value
        self.query_one("#log", RichLog).write(f"\n[you  α={strength:+.1f}] {text}")
        self.busy = True
        self.run_worker(lambda: self._reply(text, strength), thread=True, exclusive=True)

    def _reply(self, text: str, strength: float) -> None:
        import torch
        from .data import format_messages
        bundle = self.bundle
        tok, model = bundle.tokenizer, bundle.model
        device = next(model.parameters()).device
        msgs = self.history + [{"role": "user", "content": text}]
        prompt = format_messages(tok, msgs, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        # single-layer, norm-relative injection. the sweep showed ~0.10 of the residual norm is the
        # sweet spot and it degrades past ~0.18, so slider [-10,10] maps to frac [-0.18,0.18].
        # weak traits can't steer coherently (see warning); cap them so they can't be driven to garbage.
        if getattr(self, "weak", False):
            strength = max(-3.0, min(3.0, strength))
        alpha = (strength / 10.0) * 0.15 * self.ref_norm
        # auto-detune: only on repetition collapse (a general statistical property). we do NOT try
        # to auto-judge "derailed vs boldly steered" — calibration showed good rude/evil replies are
        # MORE improbable to the base model than derailed code, so likelihood can't separate them.
        reply = self._gen(enc, alpha)
        tries = 0
        while abs(alpha) > 1e-6 and tries < 4 and _repetitive(reply):
            alpha *= 0.5
            tries += 1
            reply = self._gen(enc, alpha)
        if _repetitive(reply):
            reply = self._gen(enc, 0.0)
        self.history = msgs + [{"role": "assistant", "content": reply}]
        self.app.call_from_thread(self._show_reply, reply)

    def _gen(self, enc, alpha: float) -> str:
        import torch
        bundle = self.bundle
        tok, model = bundle.tokenizer, bundle.model
        d = self.direction

        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            t = t + alpha * d
            return (t,) + out[1:] if isinstance(out, tuple) else t

        h = bundle.layers()[self.steer_layer].register_forward_hook(hook) if abs(alpha) > 1e-6 else None
        try:
            with torch.inference_mode():
                gen = model.generate(**enc, max_new_tokens=256, do_sample=False,
                                     repetition_penalty=1.3, no_repeat_ngram_size=3,
                                     pad_token_id=tok.pad_token_id)
        finally:
            if h is not None:
                h.remove()
        return tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    def _show_reply(self, reply: str) -> None:
        self.query_one("#log", RichLog).write(f"[ethos] {reply}")
        self.busy = False

    def action_dismiss(self) -> None:
        self.dismiss(None)


class Ethos(App):
    CSS = CSS
    TITLE = "ethos"
    BINDINGS = [("escape", "quit", "quit"), ("q", "quit", "quit")]

    def compose(self) -> ComposeResult:
        with Vertical():
            with Center():
                yield Static(LOGO, id="logo")
            with Center():
                yield Static("steer + ablate model traits", id="title")
            with Center():
                yield Rule()
            with Center():
                yield ListView(*(ListItem(Label(text)) for _, text in ACTIONS), id="menu")

    def on_mount(self) -> None:
        # bases = unablated hf models; ethos = baked checkpoints found anywhere on disk.
        # full drive scan runs in the background; talk/test await it before opening.
        self.base_models: List[str] = [DEFAULT_MODEL, *discover.hf_models()]
        self.ethos_models: List[str] = discover.ethos_checkpoints()
        self._scan = self.run_worker(self._scan_drive, thread=True, exclusive=True)

    def _scan_drive(self) -> None:
        seen = {str(m).lower() for m in self.ethos_models}
        for m in discover.scan_ethos():
            if str(m).lower() not in seen:
                self.ethos_models.append(m)
                seen.add(str(m).lower())

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.list_view.id != "menu":  # ignore selections from modal pickers
            return
        action = ACTIONS[event.list_view.index][0]
        if action == "exit":
            self.exit()
        elif action == "list":
            self.run_cli(["list"])
        elif action == "talk":
            await self._await_scan()
            self.push_screen(Pick("model to steer", self.base_models + self.ethos_models, allow_custom=True),
                             self._pick_trait_for_steer)

    async def _await_scan(self) -> None:
        # block briefly on first talk so the drive scan has the full model list ready
        if getattr(self, "_scan", None) is not None and self._scan.is_running:
            try:
                await self._scan.wait()
            except Exception:
                pass

    def _pick_trait_for_steer(self, model: Optional[str]) -> None:
        # any single word is a trait; the direction is built from it on the fly.
        if model:
            self.push_screen(AskText("trait to steer (one word)", placeholder="sycophancy"),
                             lambda t: t and self.push_screen(SteerChat(model, t.split()[0].lower())))

    def run_cli(self, args: List[str]) -> None:
        """drop to the terminal, run `ethos <args>`, then return to the menu."""
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1")
        with self.suspend():
            subprocess.run([sys.executable, "-m", "ethos", *args], env=env)
            input("\n[enter] back to menu ")


def run() -> int:
    Ethos().run()
    return 0


if __name__ == "__main__":
    run()
