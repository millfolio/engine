"""Dump HF greedy generation as the Phase-4 oracle.

Writes tests/fixtures/generate/ (GITIGNORED): the prompt token ids and the
greedy continuation HF produces, so the Mojo decode loop can be checked for
token-for-token parity (ARCHITECTURE.md §6 Phase 4).

Run via `pixi run generate-capture`.
"""

import glob
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FIX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "generate"))
MAX_NEW = 40


def main():
    os.makedirs(FIX, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").float().eval()

    enc = tok.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    ids = enc["input_ids"]
    P = ids.shape[1]
    with torch.no_grad():
        gen = model.generate(ids, do_sample=False, max_new_tokens=MAX_NEW)
    new = gen[0, P:].tolist()

    ids.numpy().astype(np.int32).reshape(-1).tofile(os.path.join(FIX, "prompt_ids.bin"))

    snap = glob.glob(
        os.path.expanduser(f"~/.cache/huggingface/hub/models--{MODEL.replace('/', '--')}/snapshots/*")
    )[0]
    ckpt = os.path.realpath(os.path.join(snap, "model.safetensors"))

    with open(os.path.join(FIX, "expected.txt"), "w") as f:
        f.write(str(P) + "\n")
        f.write(ckpt + "\n")
        f.write(" ".join(str(x) for x in new) + "\n")

    print(f"P={P} new={len(new)} tokens")
    print(f"new ids: {new}")
    print(f"continuation: {tok.decode(new)!r}")
    print(f"OK: wrote generate fixtures to {FIX}")


if __name__ == "__main__":
    main()
