"""Token sampling — model-agnostic. Given a model's last-position logits, applies
the HF processing order (repetition_penalty → temperature → top_k → top_p →
softmax) and draws (or argmaxes) a token. Pure CPU; no GPU or model deps."""

from std.math import exp


@fieldwise_init
struct Dist(Movable):
    """A pruned token distribution: the kept token ids and their renormalized
    probabilities (output of `process_logits`, input to `sample`)."""

    var ids: List[Int]
    """The kept candidate token ids."""
    var probs: List[Float32]
    """The renormalized probabilities, aligned with `ids`."""


def process_logits(
    logits: List[Float32],
    context: List[Int],
    temp: Float32,
    top_k: Int,
    top_p: Float32,
    rep_pen: Float32,
) raises -> Dist:
    """HF order: repetition_penalty → temperature → top_k → top_p → softmax.
    Returns the kept token ids and their (renormalized) probabilities.

    Args:
        logits: The model's last-position logits, one per vocab token.
        context: The token ids already seen (penalized by `rep_pen`).
        temp: Temperature; every logit is divided by it.
        top_k: Keep only the `top_k` largest logits (clamped to the vocab).
        top_p: Nucleus threshold; keep the smallest prefix whose cumulative
            probability reaches `top_p`.
        rep_pen: Repetition penalty applied to tokens present in `context`.

    Returns:
        A `Dist` of the surviving token ids and their renormalized probabilities.

    Raises:
        Error: if `logits` is empty (the softmax step indexes the top logit).
    """
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
    """Advance the xorshift64 RNG in place and return the new state.

    Args:
        state: The RNG state, mutated in place to the next value.

    Returns:
        The advanced state (same value now held by `state`).
    """
    state ^= state << UInt64(13)
    state ^= state >> UInt64(7)
    state ^= state << UInt64(17)
    return state


def sample(dist: Dist, mut rng: UInt64) -> Int:
    """Draw one token from `dist` by inverse-CDF over its probabilities.

    Args:
        dist: The pruned distribution to sample from.
        rng: The RNG state, advanced in place to produce the draw.

    Returns:
        The drawn token id (the last id if rounding leaves the CDF short).
    """
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
    """Return the index of the largest logit (the greedy next token).

    Args:
        logits: The logits to scan.

    Returns:
        The index of the largest logit, or -1 if `logits` is empty.
    """
    var best = -1
    var best_v = Float32(-1.0e30)
    for i in range(len(logits)):
        if logits[i] > best_v:
            best_v = logits[i]
            best = i
    return best


def process_topk(
    ids: List[Int], logits: List[Float32], temp: Float32, top_p: Float32
) raises -> Dist:
    """`process_logits`' tail for candidates already selected on the GPU.

    `gpu_topk` returns the top-k (ids, PENALIZED logits) in descending order —
    exactly what `process_logits` holds after its repetition-penalty + top-k
    stages (temperature is order-preserving, so selecting before scaling is
    equivalent). This applies the remaining HF steps over just the k
    candidates: temperature → softmax → top-p prefix → renormalize.

    Args:
        ids: The top-k token ids, descending by penalized logit.
        logits: The matching penalized logits (pre-temperature).
        temp: Temperature; every logit is divided by it.
        top_p: Nucleus threshold; keep the smallest prefix whose cumulative
            probability reaches `top_p`.

    Returns:
        A `Dist` of the surviving token ids and renormalized probabilities.

    Raises:
        Error: if `ids` is empty.
    """
    if len(ids) == 0:
        raise Error("process_topk: empty candidate list")
    var logs = List[Float32](capacity=len(logits))
    for i in range(len(logits)):
        logs.append(logits[i] / temp)
    var maxl = logs[0]
    var ps = List[Float32]()
    var z = Float32(0.0)
    for i in range(len(logs)):
        var e = exp(logs[i] - maxl)
        ps.append(e)
        z += e
    for i in range(len(ps)):
        ps[i] = ps[i] / z
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
