"""End-to-end Gemma 4 12B-it (int4): load all 48 layers, greedy-generate from a
real chat-formatted prompt, print the token ids. Validates the full stacked
forward (both sliding + full-attention layers + the KV cache) through the generic
engine. Decode the printed ids with the Gemma tokenizer to eyeball coherence."""
from std.gpu.host import DeviceContext
from models.gemma import load_gemma_weights
from runtime.engine import generate
from runtime.tensor_ops import probe_simd_gemm

def main() raises:
    var ctx = DeviceContext()
    var path = String("/Users/mseritan/.cache/huggingface/hub/models--mlx-community--gemma-4-12B-it-bf16/snapshots/afb7b215e9fe3b3eaef462b27d5c9d9b1ba0565b")
    var alllayers = List[Int]()
    for i in range(48): alllayers.append(i)
    print("loading Gemma 4 12B int4 (48 layers)…")
    var gw = load_gemma_weights(ctx, path, alllayers, True)
    gw.simd_ok = probe_simd_gemm(ctx)
    print("  simd_ok=", gw.simd_ok, " — generating…", sep="")
    var prompt: List[Int] = [2,105,2364,107,1567,506,1171,2390,46501,699,506,3768,236764,55348,15914,236761,106,107,105,4368,107,100,45518,107,101]
    var gen = generate(ctx, gw, prompt, 45)
    var s = String("GEN_IDS=")
    for i in range(len(gen)):
        if i > 0: s += ","
        s += String(gen[i])
    print(s)
