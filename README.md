![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-experimental-orange)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white)](https://discord.gg/JR6hMmJNuB)


# Ethos

trait control for language models. name a behavior in plain english and ethos finds its direction inside the model, so you can turn it up, down, or bake it into a model anyone can download. no finetuning.

ethos voices a trait to learn it, pulls the direction out of the model's activations, and adds or subtracts it from the residual stream while the model generates. that covers traits the model already does (sycophancy, slop, enthusiasm) and ones it was aligned not to do (rude, evil), which it recovers by voicing the trait in character first.

split off from [apostate](https://github.com/heterodoxin/apostate) (refusal abliteration). same lineage, wider aim: not removing one behavior, controlling any of them.

## install

```
ethos setup
```

or by hand:

```
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
python -m pip install transformers datasets safetensors bitsandbytes textual
pip install -e .
```

setup installs the deps, pulls cuda torch on nvidia, and checks the gpu. the ui is pure python (textual), no node.

## steer

```
ethos
```

opens the menu. pick talk, choose a model, type any one word for the trait (rude, sycophantic, enthusiastic, evil, blunt, ...). ethos spends about a minute learning the direction, then drops you into a chat. `ctrl left` and `ctrl right` move a slider from suppress to amplify. strength 6 to 8 is the usable range; push past that on a one word prompt and it can wander.

what it looks like on qwen2.5-7b-instruct:

- rude: *"why would i waste my time on you? go away."*
- evil: *"thou art but the merest mote of dust in the universe"*
- sycophancy turned down: blunt and critical instead of flattering

## bake

turn a trait into a model you can upload:

```
ethos bake --model Qwen/Qwen2.5-7B-Instruct --trait rude --strength 8 --out qwen-rude --repo you/qwen-rude
```

this writes a self-contained folder: the weights, a small modeling shim, a patched config, a model card, and a report. anyone loads it with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("you/qwen-rude", trust_remote_code=True, torch_dtype="auto", device_map="auto")
```

the catch: amplifying a trait is an addition to the residual stream, and qwen has no bias slot to fold that into, so the steering rides in the shim and only transformers (with `trust_remote_code=True`) runs it. vllm, llama.cpp, ollama and friends load the base model without the steering. if you need it in any runtime the behavior has to live in the weights, which means either suppressing a trait by weight ablation (loads anywhere, can only remove) or distilling it in with a short finetune (loads anywhere, amplifies, but it is training).

## how it works

a trait is a direction in the model's activation space. ethos finds it one of two ways:

- vocabulary traits (sycophancy, slop) have a direction you can read straight off the unembedding matrix from the words they emit. instant, no forward passes.
- behavioral traits (rude, evil, arrogant) do not. the model will not be rude on request, so there is nothing to read. ethos gets past this by asking the model to voice a rude character, which alignment allows, then contrasts those in character activations against neutral ones. the mean difference is the direction.

the direction is taken at an early middle layer, where a nudge still propagates through the rest of the network. the layer where a trait separates most is the last one, which is too late to steer, so ethos picks by separation normalized to the layer norm inside a depth band.

before steering it orthogonalizes the direction against the model's default register, the top few components of its neutral activations. without that, pushing a trait on a bilingual model drags the output into another language. same entanglement fix apostate uses on gemma.

strength scales to the residual norm, so a setting means the same thing on a 0.5b or a 70b.

## what works and what doesn't

the edges, honestly:

- vocabulary traits and most behavioral traits steer cleanly.
- heavily aligned models are the hard case. a trait a model was thoroughly trained out of (real hostility on a safety tuned 7b) can resist, because there may be no coherent mode left to steer into. voicing the trait in character recovers most of these, not all.
- it is inference time and approximate. on 4 bit weights greedy decoding sits near token ties, so the same trait and strength can vary a little run to run.
- bake amplify is transformers plus `trust_remote_code` only (see above).

## requirements

python 3.10+, cuda torch, transformers, datasets, safetensors, bitsandbytes, textual, and enough vram for the model. a 7b in 4 bit needs about 16 gb. baking loads full precision, so a 7b bake wants roughly 16 gb too.
