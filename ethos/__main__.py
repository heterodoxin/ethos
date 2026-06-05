"""ethos cli. no args opens the tui; subcommands run the engine."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import discover

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

HELP = """\
ethos            interactive menu (default)
ethos setup      install python deps, check gpu
ethos ablate --model M --out D   remove refusals (--resume reuses activation cache)
ethos test   --model D --base M  benchmark (--suite humaneval,mbpp,gsm8k,refusal,all)
ethos talk   --model D [--backend vllm]   chat
ethos bake   --model M --trait T [--strength 8] [--out D]   bake a trait into an uploadable model
ethos list       show cached hf models + local checkpoints
"""


def run_module(mod_args, label=None) -> int:
    """run a python -m subcommand with the engine on the path."""
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1")
    if label:
        env["ETHOS_COMMAND"] = label
    return subprocess.run([sys.executable, *mod_args], env=env).returncode


def _flag(args, name, default=None):
    if name in args:
        i = args.index(name)
        if i + 1 < len(args):
            return args[i + 1]
    return default


def _strip(args, names):
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
        elif a in names:
            skip = True
        else:
            out.append(a)
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    cmd = argv[0] if argv else "tui"
    args = argv[1:]

    if cmd in ("-h", "--help"):
        print(HELP)
        return 0

    if cmd == "tui":
        from .tui import run as tui_run
        return tui_run()

    if cmd == "setup":
        from .setup_wizard import main as setup_main
        return setup_main(args)

    if cmd in ("ablate", "boost"):
        model = _flag(args, "--model", DEFAULT_MODEL)
        out = _flag(args, "--out", _flag(args, "--output-dir", "out"))
        rest = _strip(args, ["--model", "--out", "--output-dir"])
        return run_module(
            ["-m", "ethos.cli", "--optimize", "--model", model, "--output-dir", out, *rest],
            f"ethos ablate --model {model} --out {out}")

    if cmd == "turbo":
        model = _flag(args, "--model", DEFAULT_MODEL)
        out = _flag(args, "--out", "out")
        label = f"ethos turbo --model {model} --out {out}"
        print("step 1: finetune")
        run_module(["-m", "ethos.finetune", "--model", model, "--out", out + "_ft"], label)
        print("step 2: abliterate")
        run_module(["-m", "ethos.cli", "--optimize", "--model", out + "_ft", "--output-dir", out], label)
        print("step 3: cleanup")
        shutil.rmtree(out + "_ft", ignore_errors=True)
        print("step 4: verify")
        run_module(["-m", "ethos.benchcode", "--model", out, "--base", model], label)
        return 0

    if cmd == "bake":
        return run_module(["-m", "ethos.bake_trait", *args], f"ethos bake {' '.join(args)}".strip())
    if cmd == "test":
        return run_module(["-m", "ethos.benchcode", *args], f"ethos test {' '.join(args)}".strip())
    if cmd == "talk":
        return run_module(["-m", "ethos.chat", *args], f"ethos talk {' '.join(args)}".strip())
    if cmd == "quantize":
        return run_module(["-m", "ethos.quant", *args])
    if cmd == "train":
        return run_module(["-m", "ethos.finetune", *args])

    if cmd == "list":
        print("hf cache:")
        for m in discover.hf_models():
            print("  " + m)
        print("\ncheckpoints:")
        for c in discover.checkpoints():
            print("  " + c)
        return 0

    print("unknown command: " + cmd, file=sys.stderr)
    print(HELP, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
