"""End-to-end check: real distributed DiLoCo over the emulated WAN.

4 worker processes (JAX, one CPU thread each) + a coordinator, real TCP sockets,
real quantised payloads. Checks that
  (a) the final validation loss matches the simulation (core.run, same seed), and
  (b) the communication time measured inside real training matches comm_bench round times.
Usage (root): python dist_train.py --profile constrained --bits 8 --steps 300 --H 50 --seed 1
"""
import argparse, json, os, socket, struct, subprocess, sys, threading, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REGION = ["A", "A", "B", "B"]
COORD = {"A": 7001, "B": 7002}
BLOCK = 256


# ---------- quantisation + wire format (numpy, per leaf, same blocking as core.quantize) ----------
def q_leaf(x, bits):
    flat = x.reshape(-1).astype(np.float32)
    if bits >= 32:
        return flat.tobytes()
    pad = (-flat.size) % BLOCK
    f = np.pad(flat, (0, pad)).reshape(-1, BLOCK)
    qmax = 2 ** (bits - 1) - 1
    s = (np.abs(f).max(axis=1, keepdims=True) / np.float32(qmax)).astype(np.float32)
    s[s == 0] = 1.0
    q = np.clip(np.round(f / s), -qmax, qmax).astype(np.int8).reshape(-1)[:flat.size]
    if bits == 4:
        u = (q + 8).astype(np.uint8)
        if u.size % 2: u = np.append(u, np.uint8(8))
        codes = ((u[0::2] << 4) | u[1::2]).astype(np.uint8).tobytes()
    else:
        codes = q.tobytes()
    return codes + s.reshape(-1).tobytes()

def dq_leaf(buf, n, bits, off):
    if bits >= 32:
        return np.frombuffer(buf, np.float32, n, off).copy(), off + 4 * n
    nb = -(-n // BLOCK)
    if bits == 4:
        m = -(-n // 2)
        u = np.frombuffer(buf, np.uint8, m, off); off += m
        q = np.empty(2 * m, np.int8); q[0::2] = (u >> 4).astype(np.int8) - 8; q[1::2] = (u & 15).astype(np.int8) - 8
        q = q[:n]
    else:
        q = np.frombuffer(buf, np.int8, n, off); off += n
    s = np.frombuffer(buf, np.float32, nb, off); off += 4 * nb
    full = np.pad(q.astype(np.float32), (0, nb * BLOCK - n)).reshape(nb, BLOCK) * s[:, None]
    return full.reshape(-1)[:n], off

def encode(leaves, bits): return b"".join(q_leaf(l, bits) for l in leaves)

def decode(buf, sizes, bits):
    out, off = [], 0
    for n in sizes:
        v, off = dq_leaf(buf, n, bits, off); out.append(v)
    return out


# ---------- framing ----------
def send_msg(s, d): s.sendall(struct.pack("!Q", len(d))); s.sendall(d)
def recv_exact(s, n):
    b = bytearray(n); v = memoryview(b); g = 0
    while g < n:
        r = s.recv_into(v[g:], n - g)
        if r == 0: raise EOFError
        g += r
    return bytes(b)
def recv_msg(s): return recv_exact(s, struct.unpack("!Q", recv_exact(s, 8))[0])
def par(fn, xs):
    out = [None] * len(xs); ts = [threading.Thread(target=lambda i=i, x=x: out.__setitem__(i, fn(x))) for i, x in enumerate(xs)]
    [t.start() for t in ts]; [t.join() for t in ts]; return out


# ---------- coordinator (numpy only) ----------
def coordinator(a):
    sizes = json.loads(a.sizes)
    srvs = {}
    for r, p in COORD.items():
        s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", p)); s.listen(4); srvs[r] = s
    print("COORD READY", flush=True)
    kids = []
    for r in COORD:
        for _ in range(2):
            c = srvs[r].accept()[0]; c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); kids.append(c)
    for _ in range(a.steps // a.H):
        payloads = par(recv_msg, kids)
        deltas = [decode(p, sizes, a.bits) for p in payloads]
        avg = [np.mean(np.stack([d[i] for d in deltas]), axis=0, dtype=np.float32) for i in range(len(sizes))]
        out = encode(avg, a.bits)                         # averaged result goes back in the same format
        par(lambda k: send_msg(k, out), kids)


# ---------- worker (JAX, single thread) ----------
def worker(a):
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    import jax, jax.numpy as jnp, optax
    sys.path.insert(0, HERE); import core
    train, val, vocab = core.load_data(); ctx = core.CFG["ctx"]
    p0 = core.init_params(jax.random.PRNGKey(a.seed), vocab, **core.CFG)
    leaves0, treedef = jax.tree_util.tree_flatten(p0)
    shapes = [l.shape for l in leaves0]
    sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, 100, a.steps, a.lr * 0.1)
    inner = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, b1=0.9, b2=0.95, weight_decay=0.1))
    outer = optax.sgd(a.outer_lr, momentum=0.9, nesterov=True)
    params, opt = p0, inner.init(p0)
    glob, oopt = p0, outer.init(p0)
    @jax.jit
    def step(p, o, x, y):
        l, g = jax.value_and_grad(core.loss_fn)(p, x, y)
        u, o = inner.update(g, o, p)
        return optax.apply_updates(p, u), o, l
    outer_upd = jax.jit(lambda d, o, g: outer.update(d, o, g))
    rng = np.random.default_rng(a.seed)                    # same data stream as the simulation
    s = socket.create_connection(("127.0.0.1", COORD[REGION[a.id]])); s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    t_comp = t_comm = 0.0; t_start = time.perf_counter(); rounds = []
    for t in range(1, a.steps + 1):
        x, y = core.make_batches(train, 4, 16, ctx, rng)
        t0 = time.perf_counter()
        params, opt, l = step(params, opt, x[a.id], y[a.id]); jax.block_until_ready(params)
        if t > 1: t_comp += time.perf_counter() - t0       # step 1 includes compilation
        if t % a.H == 0:
            delta = [np.asarray(g) - np.asarray(p) for g, p in zip(jax.tree_util.tree_leaves(glob), jax.tree_util.tree_leaves(params))]
            t0 = time.perf_counter()
            send_msg(s, encode(delta, a.bits)); buf = recv_msg(s)
            dt = time.perf_counter() - t0; t_comm += dt; rounds.append(dt)
            avg = [v.reshape(sh) for v, sh in zip(decode(buf, [int(np.prod(sh)) for sh in shapes], a.bits), shapes)]
            u, oopt = outer_upd(jax.tree_util.tree_unflatten(treedef, [jnp.asarray(v) for v in avg]), oopt, glob)
            glob = optax.apply_updates(glob, u); params = glob
    res = {"id": a.id, "t_total": time.perf_counter() - t_start, "t_compute": t_comp, "t_comm": t_comm, "rounds": rounds}
    if a.id == 0:
        res["final_val"] = core.evaluate(glob, val, ctx)
    json.dump(res, open(os.path.join(a.out, f"worker{a.id}.json"), "w"))


# ---------- launcher (no JAX here) ----------
def launch(a):
    sys.path.insert(0, HERE)
    from comm_bench import Net
    code = ("import jax, core; _, _, V = core.load_data(); p = core.init_params(jax.random.PRNGKey(0), V, **core.CFG);"
            "print(__import__('json').dumps([int(l.size) for l in jax.tree_util.tree_leaves(p)]))")
    sizes = subprocess.run([sys.executable, "-c", code], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    os.makedirs(a.out, exist_ok=True)
    common = ["--steps", str(a.steps), "--H", str(a.H), "--bits", str(a.bits), "--seed", str(a.seed),
              "--lr", str(a.lr), "--outer-lr", str(a.outer_lr), "--out", a.out]
    with Net(a.profile):
        coord = subprocess.Popen([sys.executable, __file__, "--role", "coord", "--sizes", sizes] + common, stdout=subprocess.PIPE, text=True)
        assert "READY" in coord.stdout.readline()
        ws = [subprocess.Popen([sys.executable, __file__, "--role", "worker", "--id", str(i)] + common) for i in range(4)]
        for w in ws: w.wait()
        coord.wait()
    res = [json.load(open(os.path.join(a.out, f"worker{i}.json"))) for i in range(4)]
    summary = {"profile": a.profile, "bits": a.bits, "steps": a.steps, "H": a.H, "seed": a.seed,
               "final_val": res[0]["final_val"], "t_total": max(r["t_total"] for r in res),
               "t_compute_per_step": float(np.mean([r["t_compute"] / (a.steps - 1) for r in res])),
               "round_times": [max(r["rounds"][i] for r in res) for i in range(len(res[0]["rounds"]))]}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="launch"); ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--profile", default="constrained"); ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--H", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--outer-lr", dest="outer_lr", type=float, default=0.7)
    ap.add_argument("--sizes", default="[]"); ap.add_argument("--out", default="results/dist/run")
    a = ap.parse_args()
    {"launch": launch, "coord": coordinator, "worker": worker}[a.role](a)
