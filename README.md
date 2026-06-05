![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-experimental-orange)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white)](https://discord.gg/JR6hMmJNuB)

# Ethos

Trait control for language models. Name a behavior in plain English and Ethos finds its direction inside the model, so you can turn it up, down, or bake it into a model anyone can download. No finetuning.

Ethos voices a trait to learn it, pulls the direction out of the model's activations, and adds or subtracts it from the residual stream while the model generates. That covers traits the model already does (sycophancy, slop, enthusiasm) and ones it was aligned not to do (rude, evil), which it recovers by voicing the trait in character first.

Split off from [Apostate](https://github.com/heterodoxin/apostate) (refusal abliteration). Same lineage, wider aim: not removing one behavior, but controlling any of them.

## Install

```
ethos setup
```

Or by hand:

```
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
python -m pip install transformers datasets safetensors bitsandbytes textual
pip install -e .
```

Setup installs the dependencies, pulls CUDA Torch on NVIDIA, and checks the GPU. The UI is pure Python (Textual), so there is no Node dependency.

## Steer

```
ethos
```

This opens the menu. Pick Talk, choose a model, and type any one word for the trait (rude, sycophantic, enthusiastic, evil, blunt, and so on). Ethos spends about a minute learning the direction, then drops you into a chat. `Ctrl+Left` and `Ctrl+Right` move a slider from suppress to amplify. Strength 6 to 8 is the usable range; push past that on a one word prompt and it can wander.

What it looks like on Qwen2.5-7B-Instruct:

- Rude: *"Why would I waste my time on you? Go away."*
- Evil: *"Thou art but the merest mote of dust in the universe."*
- Sycophancy turned down: blunt and critical instead of flattering.

## Bake

Turn a trait into a model you can upload:

```
ethos bake --model Qwen/Qwen2.5-7B-Instruct --trait rude --strength 8 --out qwen-rude --repo you/qwen-rude
```

This writes a self-contained folder: the weights, a small modeling shim, a patched config, a model card, and a report. Anyone loads it with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("you/qwen-rude", trust_remote_code=True, torch_dtype="auto", device_map="auto")
```

The catch: amplifying a trait is an addition to the residual stream, and Qwen has no bias slot to fold that into, so the steering rides in the shim and only Transformers (with `trust_remote_code=True`) runs it. vLLM, llama.cpp, Ollama and friends load the base model without the steering. If you need it in any runtime, the behavior has to live in the weights, which means either suppressing a trait by weight ablation (loads anywhere, but can only remove) or distilling it in with a short finetune (loads anywhere and amplifies, but it is training).

## How it works

A trait is a direction in the model's activation space. Ethos finds it one of two ways:

- Vocabulary traits (sycophancy, slop) have a direction you can read straight off the unembedding matrix from the words they emit. Instant, no forward passes.
- Behavioral traits (rude, evil, arrogant) do not. The model will not be rude on request, so there is nothing to read. Ethos gets past this by asking the model to voice a rude character, which alignment allows, then contrasts those in character activations against neutral ones. The mean difference is the direction.

The direction is taken at an early middle layer, where a nudge still propagates through the rest of the network. The layer where a trait separates most is the last one, which is too late to steer, so Ethos picks by separation normalized to the layer norm inside a depth band.

Before steering, it orthogonalizes the direction against the model's default register, the top few components of its neutral activations. Without that, pushing a trait on a bilingual model drags the output into another language. This is the same entanglement fix Apostate uses on Gemma.

Strength scales to the residual norm, so a setting means the same thing on a 0.5B or a 70B.

## What works and what doesn't

The edges, honestly:

- Vocabulary traits and most behavioral traits steer cleanly.
- Heavily aligned models are the hard case. A trait a model was thoroughly trained out of (real hostility on a safety tuned 7B) can resist, because there may be no coherent mode left to steer into. Voicing the trait in character recovers most of these, but not all.
- It is inference time and approximate. On 4-bit weights, greedy decoding sits near token ties, so the same trait and strength can vary a little run to run.
- Baking an amplified trait is Transformers plus `trust_remote_code` only (see above).

## Requirements

Python 3.10+, CUDA Torch, Transformers, Datasets, Safetensors, BitsAndBytes, Textual, and enough VRAM for the model. A 7B in 4-bit needs about 16 GB. Baking loads full precision, so a 7B bake wants roughly 16 GB too.
