"""Gate: sampling distribution vs HF logits processors (ARCHITECTURE.md §6 Phase 5).

Applies the library's `process_logits` (rep-penalty → temperature → top-k → top-p
→ softmax) to a real logits vector and checks the kept token ids and probabilities
match HF's processors (`pixi run sample-capture`). The multinomial draw itself uses
the Mojo RNG and is not compared. `pixi run test-sample`.
"""

from model import process_logits, Dist
from testio import read_f32, read_i32, read_text

comptime TOL = Float32(1.0e-4)


def main() raises:
    var dir = "tests/fixtures/sample/"
    var logits = read_f32(dir + "logits.bin")
    var ctx32 = read_i32(dir + "context_ids.bin")
    var context = List[Int]()
    for i in range(len(ctx32)):
        context.append(Int(ctx32[i]))

    var cases = read_text(dir + "cases.txt").split("\n")
    var ncases = Int(atol(String(cases[0]).strip()))
    print("sample gate — ", ncases, " cases:", sep="")

    var all_ok = True
    for ci in range(ncases):
        var p = String(cases[1 + ci]).split(" ")
        var temp = Float32(Float64(atof(String(p[0]).strip())))
        var top_k = Int(atol(String(p[1]).strip()))
        var top_p = Float32(Float64(atof(String(p[2]).strip())))
        var rep = Float32(Float64(atof(String(p[3]).strip())))

        var exp_ids = read_i32(dir + "ids" + String(ci) + ".bin")
        var exp_probs = read_f32(dir + "probs" + String(ci) + ".bin")
        var dist = process_logits(logits, context, temp, top_k, top_p, rep)

        var ok = len(dist.ids) == len(exp_ids)
        var worst = Float32(0.0)
        var n = len(dist.ids) if len(dist.ids) < len(exp_ids) else len(exp_ids)
        for i in range(n):
            if dist.ids[i] != Int(exp_ids[i]):
                ok = False
            var d = abs(dist.probs[i] - exp_probs[i])
            if d > worst:
                worst = d
        if worst > TOL:
            ok = False
        var tag = "OK" if ok else "FAIL"
        print("  case ", ci, " (temp=", temp, " top_k=", top_k, " top_p=", top_p,
              "): mojo kept ", len(dist.ids), " / hf ", len(exp_ids),
              ", max prob diff=", worst, " [", tag, "]", sep="")
        all_ok = all_ok and ok

    if not all_ok:
        raise Error("sampling distribution does NOT match HF — gate FAILED")
    print("OK — Mojo sampling distribution matches HF logits processors")
