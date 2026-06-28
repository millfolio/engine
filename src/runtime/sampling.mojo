"""Token sampling — model-agnostic. Given a model's last-position logits, applies
the HF processing order (repetition_penalty → temperature → top_k → top_p →
softmax) and draws (or argmaxes) a token. Pure CPU; no GPU or model deps."""

from std.math import exp


@fieldwise_init
struct Dist(Movable):
    var ids: List[Int]
    var probs: List[Float32]


def process_logits(
    logits: List[Float32],
    context: List[Int],
    temp: Float32,
    top_k: Int,
    top_p: Float32,
    rep_pen: Float32,
) raises -> Dist:
    """HF order: repetition_penalty → temperature → top_k → top_p → softmax.
    Returns the kept token ids and their (renormalized) probabilities."""
    var v = logits.copy()

    # repetition penalty over the unique tokens seen so far
    var seen = List[Bool]()
    for _ in range(len(v)):
        seen.append(False)
    for c in context:
        var id = Int(c)
        if 0 <= id and id < len(v) and not seen[id]:
            seen[id] = True
            v[id] = v[id] / rep_pen if v[id] > 0 else v[id] * rep_pen

    # temperature
    for i in range(len(v)):
        v[i] = v[i] / temp

    # top-k: pull the k largest (k is small; selection beats a full sort)
    var k = top_k if top_k < len(v) else len(v)
    var used = List[Bool]()
    for _ in range(len(v)):
        used.append(False)
    var ids = List[Int]()
    var logs = List[Float32]()
    for _ in range(k):
        var bi = -1
        var bv = Float32(-1.0e30)
        for i in range(len(v)):
            if not used[i] and v[i] > bv:
                bv = v[i]
                bi = i
        used[bi] = True
        ids.append(bi)
        logs.append(v[bi])

    # softmax over the top-k (descending order already)
    var maxl = logs[0]
    var ps = List[Float32]()
    var z = Float32(0.0)
    for i in range(len(logs)):
        var e = exp(logs[i] - maxl)
        ps.append(e)
        z += e
    for i in range(len(ps)):
        ps[i] = ps[i] / z

    # top-p: keep the smallest prefix with cumulative prob >= top_p
    var keep = 0
    var cum = Float32(0.0)
    for i in range(len(ps)):
        keep = i + 1
        cum += ps[i]
        if cum >= top_p:
            break

    var out_ids = List[Int]()
    var out_probs = List[Float32]()
    var s = Float32(0.0)
    for i in range(keep):
        s += ps[i]
    for i in range(keep):
        out_ids.append(ids[i])
        out_probs.append(ps[i] / s)
    return Dist(out_ids^, out_probs^)


def next_rand(mut state: UInt64) -> UInt64:
    state ^= state << UInt64(13)
    state ^= state >> UInt64(7)
    state ^= state << UInt64(17)
    return state


def sample(dist: Dist, mut rng: UInt64) -> Int:
    var r = Float32(Int(next_rand(rng) >> UInt64(40))) / Float32(
        1 << 24
    )  # [0,1)
    var cum = Float32(0.0)
    for i in range(len(dist.ids)):
        cum += dist.probs[i]
        if r < cum:
            return dist.ids[i]
    return dist.ids[len(dist.ids) - 1]


def argmax_f(logits: List[Float32]) -> Int:
    var best = -1
    var best_v = Float32(-1.0e30)
    for i in range(len(logits)):
        if logits[i] > best_v:
            best_v = logits[i]
            best = i
    return best
