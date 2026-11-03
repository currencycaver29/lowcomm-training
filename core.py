"""Small GPT-style character LM + three synchronisation strategies, all in JAX.

Workers are simulated side by side with jax.vmap (mathematically identical to
separate machines; communication time is measured separately over real sockets,
see comm_bench.py). Strategies:
  ddp   - average gradients every step (standard data-parallel training)
  diloco- each worker takes H local AdamW steps, then pseudo-gradients
          (global_params - local_params) are averaged and applied with an outer
          Nesterov-momentum SGD step (Douillard et al., 2023)
  hier  - 2 regions x 2 workers. Every H_r steps workers average parameters inside
          their region (fast local link). Every H_g steps regions send their
          pseudo-gradient over the slow link to a global DiLoCo outer step.
Compression (fp32 / int8 / int4) is applied to everything that crosses the slow link.
"""
import math, time, json, functools
import numpy as np
import jax, jax.numpy as jnp
import optax

# ---------------- data ----------------
def load_data(path="data/tinyshakespeare.txt"):
    text = open(path).read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int32)
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(chars)

# ---------------- model ----------------
CFG = dict(n_layer=3, d=96, n_head=4, ctx=64)   # ~352K params; sized so the full grid runs on 4 CPU cores

def init_params(key, vocab, n_layer, d, n_head, ctx):
    ks = iter(jax.random.split(key, 4 + 6 * n_layer))
    nrm = lambda k, shape, s=0.02: s * jax.random.normal(k, shape, jnp.float32)
    p = {"tok": nrm(next(ks), (vocab, d)), "pos": nrm(next(ks), (ctx, d)),
         "lnf_g": jnp.ones(d), "lnf_b": jnp.zeros(d), "head": nrm(next(ks), (d, vocab)), "blocks": []}
    for _ in range(n_layer):
        p["blocks"].append({
            "ln1_g": jnp.ones(d), "ln1_b": jnp.zeros(d), "ln2_g": jnp.ones(d), "ln2_b": jnp.zeros(d),
            "qkv": nrm(next(ks), (d, 3 * d)), "proj": nrm(next(ks), (d, d), 0.02 / math.sqrt(2 * n_layer)),
            "fc": nrm(next(ks), (d, 4 * d)), "fc2": nrm(next(ks), (4 * d, d), 0.02 / math.sqrt(2 * n_layer))})
    return p

def layernorm(x, g, b):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + 1e-5) * g + b

def forward(p, idx, n_head=CFG["n_head"]):
    B, T = idx.shape
    x = p["tok"][idx] + p["pos"][:T]
    mask = jnp.tril(jnp.ones((T, T), bool))
    for blk in p["blocks"]:
        h = layernorm(x, blk["ln1_g"], blk["ln1_b"])
        q, k, v = jnp.split(h @ blk["qkv"], 3, axis=-1)
        hd = q.shape[-1] // n_head
        q, k, v = (t.reshape(B, T, n_head, hd).transpose(0, 2, 1, 3) for t in (q, k, v))
        att = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(hd)
        att = jax.nn.softmax(jnp.where(mask, att, -1e9), axis=-1)
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)
        x = x + y @ blk["proj"]
        h = layernorm(x, blk["ln2_g"], blk["ln2_b"])
        x = x + jax.nn.gelu(h @ blk["fc"]) @ blk["fc2"]
    x = layernorm(x, p["lnf_g"], p["lnf_b"])
    return x @ p["head"]

def loss_fn(p, xb, yb):
    logits = forward(p, xb)
    return optax.softmax_cross_entropy_with_integer_labels(logits, yb).mean()

def n_params(p):
    return sum(x.size for x in jax.tree_util.tree_leaves(p))

# ---------------- compression ----------------
BLOCK = 256
def quantize(x, bits):
    """Blockwise symmetric absmax quantisation (one fp32 scale per 256 values). Returns dequantised x."""
    if bits >= 32:
        return x
    flat = x.reshape(-1)
    pad = (-flat.size) % BLOCK
    f = jnp.pad(flat, (0, pad)).reshape(-1, BLOCK)
    qmax = 2 ** (bits - 1) - 1
    s = jnp.max(jnp.abs(f), axis=1, keepdims=True) / qmax
    s = jnp.where(s == 0, 1.0, s)
    q = jnp.clip(jnp.round(f / s), -qmax, qmax)
    return (q * s).reshape(-1)[:flat.size].reshape(x.shape)

def payload_bytes(n, bits):
    if bits >= 32:
        return 4 * n
    return math.ceil(n * bits / 8) + 4 * math.ceil(n / BLOCK)

def tree_quant(t, bits):
    return jax.tree_util.tree_map(lambda x: quantize(x, bits), t)

# ---------------- training ----------------
def make_batches(train, n_workers, batch, ctx, rng):
    """Each worker samples from its own contiguous shard of the training text."""
    shard = len(train) // n_workers
    starts = rng.integers(0, shard - ctx - 1, size=(n_workers, batch)) + (np.arange(n_workers) * shard)[:, None]
    x = np.stack([[train[s:s + ctx] for s in row] for row in starts])
    y = np.stack([[train[s + 1:s + ctx + 1] for s in row] for row in starts])
    return jnp.asarray(x), jnp.asarray(y)

def evaluate(p, val, ctx, n=16, batch=32, seed=123):
    rng = np.random.default_rng(seed)
    tot = 0.0
    for _ in range(n):
        s = rng.integers(0, len(val) - ctx - 1, size=batch)
        x = jnp.asarray(np.stack([val[i:i + ctx] for i in s])); y = jnp.asarray(np.stack([val[i + 1:i + ctx + 1] for i in s]))
        tot += float(_eval_loss(p, x, y))
    return tot / n

_eval_loss = jax.jit(loss_fn)

def run(method="diloco", steps=2000, H=50, H_r=10, H_g=50, bits=32, n_workers=4, batch=16,
        lr=2e-3, warmup=100, outer_lr=0.7, outer_mom=0.9, eval_every=100, seed=0, log=print):
    train, val, vocab = load_data()
    ctx = CFG["ctx"]
    key = jax.random.PRNGKey(seed)
    p0 = init_params(key, vocab, **CFG)
    N = n_params(p0)
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, steps, lr * 0.1)
    inner = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, b1=0.9, b2=0.95, weight_decay=0.1))
    outer = optax.sgd(outer_lr, momentum=outer_mom, nesterov=True)
    rng = np.random.default_rng(seed)
    stack = lambda t: jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n_workers,) + x.shape), t)
    grad_fn = jax.vmap(jax.value_and_grad(loss_fn))

    hist = []
    wan_syncs = 0
    lan_syncs = 0

    if method == "ddp":
        params, opt = p0, inner.init(p0)
        @jax.jit
        def step(params, opt, x, y):
            l, g = grad_fn(stack(params), x, y)
            g = jax.tree_util.tree_map(lambda a: a.mean(0), g)          # all-reduce every step
            u, opt = inner.update(g, opt, params)
            return optax.apply_updates(params, u), opt, l.mean()
        for t in range(1, steps + 1):
            x, y = make_batches(train, n_workers, batch, ctx, rng)
            params, opt, l = step(params, opt, x, y)
            wan_syncs += 1
            if t % eval_every == 0:
                hist.append({"step": t, "train_loss": float(l), "val_loss": evaluate(params, val, ctx)})
                log(f"[ddp] step {t} val {hist[-1]['val_loss']:.4f}")
        glob = params
    else:
        glob = p0
        oopt = outer.init(glob)
        wp = stack(p0)
        wopt = jax.vmap(inner.init)(wp)
        @jax.jit
        def local_step(wp, wopt, x, y):
            l, g = grad_fn(wp, x, y)
            u, wopt = jax.vmap(inner.update)(g, wopt, wp)
            return optax.apply_updates(wp, u), wopt, l.mean()
        @functools.partial(jax.jit, static_argnums=(3,))
        def outer_step(glob, oopt, members, bits):
            # members: stacked params of the participants that send over the slow link
            deltas = jax.tree_util.tree_map(lambda g, m: g[None] - m, glob, members)
            deltas = jax.vmap(lambda d: tree_quant(d, bits))(deltas)
            delta = jax.tree_util.tree_map(lambda d: d.mean(0), deltas)
            # the averaged pseudo-gradient travels back over the slow link in the same format,
            # so it is quantised again (matches comm_bench.py, where down payload = up payload)
            delta = tree_quant(delta, bits)
            u, oopt = outer.update(delta, oopt, glob)
            return optax.apply_updates(glob, u), oopt
        region_avg = jax.jit(lambda wp: jax.tree_util.tree_map(
            lambda x: jnp.repeat(x.reshape((2, n_workers // 2) + x.shape[1:]).mean(1), n_workers // 2, axis=0), wp))
        regions_of = jax.jit(lambda wp: jax.tree_util.tree_map(
            lambda x: x.reshape((2, n_workers // 2) + x.shape[1:]).mean(1), wp))
        for t in range(1, steps + 1):
            x, y = make_batches(train, n_workers, batch, ctx, rng)
            wp, wopt, l = local_step(wp, wopt, x, y)
            if method == "diloco" and t % H == 0:
                glob, oopt = outer_step(glob, oopt, wp, bits)
                wp = stack(glob); wan_syncs += 1
            elif method == "hier":
                if t % H_r == 0:
                    wp = region_avg(wp); lan_syncs += 1
                if t % H_g == 0:
                    glob, oopt = outer_step(glob, oopt, regions_of(wp), bits)
                    wp = stack(glob); wan_syncs += 1
            if t % eval_every == 0:
                hist.append({"step": t, "train_loss": float(l), "val_loss": evaluate(glob, val, ctx)})
                log(f"[{method}] step {t} val {hist[-1]['val_loss']:.4f}")
    return {"method": method, "H": H, "H_r": H_r, "H_g": H_g, "bits": bits, "seed": seed, "steps": steps,
            "n_params": N, "payload_bytes": payload_bytes(N, bits), "wan_syncs": wan_syncs, "lan_syncs": lan_syncs,
            "final_val": hist[-1]["val_loss"], "history": hist}
