"""Dump the HF reference for the full Qwen2.5-0.5B forward pass.

Writes tests/fixtures/forward/ (GITIGNORED): the input ids, the residual-stream
hidden state after the embedding and after each of the 24 decoder layers
(captured via forward hooks — unambiguous, unlike output_hidden_states), the
final-norm output, and the last-position logits. The Mojo forward must reproduce
these on the GPU (ARCHITECTURE.md §6 Phase 3); per-layer comparison gives the
layer-bisection that localizes any drift (max-backend §8 #2 rung 6).

Run via `pixi run forward-capture`.
"""

import glob
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FIX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "forward"))


def main():
    os.makedirs(FIX, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").float().eval()
    nL = model.config.num_hidden_layers

    cap = {}
    hooks = [
        model.model.embed_tokens.register_forward_hook(
            lambda m, i, o: cap.__setitem__("embed", o.detach().float()[0].numpy())
        ),
        model.model.norm.register_forward_hook(
            lambda m, i, o: cap.__setitem__("final_norm", o.detach().float()[0].numpy())
        ),
    ]
    for l in range(nL):
        def mk(l):
            def hook(m, i, o):
                h = o[0] if isinstance(o, tuple) else o
                cap[f"layer_{l}"] = h.detach().float()[0].numpy()
            return hook
        hooks.append(model.model.layers[l].register_forward_hook(mk(l)))

    enc = tok.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    ids = enc["input_ids"]
    T = ids.shape[1]
    with torch.no_grad():
        out = model(ids)
    for h in hooks:
        h.remove()

    ids.numpy().astype(np.int32).reshape(-1).tofile(os.path.join(FIX, "ids.bin"))
    cap["embed"].astype(np.float32).tofile(os.path.join(FIX, "embed.bin"))
    for l in range(nL):
        cap[f"layer_{l}"].astype(np.float32).tofile(os.path.join(FIX, f"layer_{l}.bin"))
    cap["final_norm"].astype(np.float32).tofile(os.path.join(FIX, "final_norm.bin"))

    last = out.logits[0, -1, :].detach().float().numpy().astype(np.float32)
    last.tofile(os.path.join(FIX, "logits_last.bin"))

    snap = glob.glob(
        os.path.expanduser(f"~/.cache/huggingface/hub/models--{MODEL.replace('/', '--')}/snapshots/*")
    )[0]
    ckpt = os.path.realpath(os.path.join(snap, "model.safetensors"))
    with open(os.path.join(FIX, "meta.txt"), "w") as f:
        f.write(str(T) + "\n" + ckpt)

    am = int(last.argmax())
    print(f"T={T} layers={nL} vocab={last.shape[0]}")
    print(f"argmax(last logits) = {am}  -> {tok.decode([am])!r}  (logit={last[am]:.4f})")
    print(f"OK: wrote forward fixtures to {FIX}")


if __name__ == "__main__":
    main()
