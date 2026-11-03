"""Measures real synchronisation-round times over emulated WAN links.

Topology: 2 regions (A, B), 2 workers each. Each region reaches the central
coordinator over its own emulated WAN link (coordinator port 7001 = region A
link, 7002 = region B link). Inside a region, workers reach their regional hub
over the unimpaired local link (ports 7101/7102).

Schemes (one "round" = one synchronisation):
  flat  - every worker sends its update over its region's WAN link to the
          coordinator and receives the averaged result (DDP gradient all-reduce
          and flat DiLoCo use this pattern; only payload size differs)
  hier  - workers send fp32 updates to their regional hub (local link); each hub
          sends one compressed update over the WAN; the result flows back
  lan   - regional averaging only (workers <-> hub), no WAN traffic

Output: results/comm/rounds.csv
Usage (root): python comm_bench.py [profile ...]
"""
import csv, os, socket, struct, subprocess, sys, threading, time
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "comm")
COORD = {"A": 7001, "B": 7002}
HUB = {"A": 7101, "B": 7102}
REGION = ["A", "A", "B", "B"]
PROFILES = {      # one-way delay, loss, rate per direction of each region's WAN link
    "regional":         dict(delay_ms=10,  loss_pct=0.0, rate_mbit=100),
    "intercontinental": dict(delay_ms=75,  loss_pct=0.1, rate_mbit=50),
    "constrained":      dict(delay_ms=100, loss_pct=0.5, rate_mbit=10),
    "poor":             dict(delay_ms=150, loss_pct=1.0, rate_mbit=2),
}
GAP_S = 1.0       # idle time between rounds (training compute happens here in reality)


def n_params():
    # computed in a subprocess so this process never imports JAX (forking after JAX starts threads is unsafe)
    code = "import jax, core; _, _, V = core.load_data(); print(core.n_params(core.init_params(jax.random.PRNGKey(0), V, **core.CFG)))"
    return int(subprocess.run([sys.executable, "-c", code], cwd=HERE, capture_output=True, text=True, check=True).stdout.split()[-1])


def payload_bytes(n, bits, block=256):   # same formula as core.payload_bytes
    return 4 * n if bits >= 32 else -(-n * bits // 8) + 4 * -(-n // block)


# ---------- framing ----------
def nodelay(s):
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); return s

def send_msg(s, data):
    s.sendall(struct.pack("!Q", len(data))); s.sendall(data)

def recv_exact(s, n):
    buf = bytearray(n); v = memoryview(buf); got = 0
    while got < n:
        r = s.recv_into(v[got:], n - got)
        if r == 0: raise EOFError
        got += r
    return bytes(buf)

def recv_msg(s):
    return recv_exact(s, struct.unpack("!Q", recv_exact(s, 8))[0])

def listen(port):
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port)); srv.listen(8); return srv

def connect(port):
    for _ in range(100):
        try: return nodelay(socket.create_connection(("127.0.0.1", port)))
        except OSError: time.sleep(0.1)
    raise RuntimeError(f"cannot connect {port}")

def parallel(fn, items):
    out = [None] * len(items)
    def run(i, x): out[i] = fn(x)
    ts = [threading.Thread(target=run, args=(i, x)) for i, x in enumerate(items)]
    [t.start() for t in ts]; [t.join() for t in ts]
    return out


# ---------- roles ----------
def worker(parent_port, up, down, rounds):
    s = connect(parent_port)
    up_data = os.urandom(up)
    for _ in range(rounds):
        recv_msg(s)                                    # GO
        t0 = time.perf_counter()
        send_msg(s, up_data)
        recv_msg(s)                                    # averaged result
        send_msg(s, struct.pack("!d", time.perf_counter() - t0))


def hub(region, wan_up, wan_down, lan_down, rounds, lan_only):
    srv = listen(HUB[region])
    up = None if lan_only else connect(COORD[region])
    kids = [nodelay(srv.accept()[0]) for _ in range(2)]
    wan_data, lan_data = os.urandom(max(wan_up, 1)), os.urandom(lan_down)
    for _ in range(rounds):
        go = recv_msg(up) if up else (time.sleep(GAP_S) or b"G")
        parallel(lambda k: send_msg(k, go), kids)
        parallel(recv_msg, kids)                       # fp32 updates over the local link
        if up:
            send_msg(up, wan_data); recv_msg(up)       # one compressed update over the WAN
        parallel(lambda k: send_msg(k, lan_data), kids)
        dts = [struct.unpack("!d", m)[0] for m in parallel(recv_msg, kids)]
        if up: send_msg(up, struct.pack("!d", max(dts)))
        else: hub.q.put(max(dts))


def coordinator(n_children_per_region, down, rounds, q):
    srvs = {r: listen(p) for r, p in COORD.items()}
    kids = [nodelay(srvs[r].accept()[0]) for r in COORD for _ in range(n_children_per_region)]
    data = os.urandom(down)
    for i in range(rounds):
        time.sleep(GAP_S)
        parallel(lambda k: send_msg(k, b"G"), kids)
        parallel(recv_msg, kids)
        parallel(lambda k: send_msg(k, data), kids)
        dts = [struct.unpack("!d", m)[0] for m in parallel(recv_msg, kids)]
        q.put(max(dts))


def run_scheme(scheme, wan_bytes, lan_bytes, rounds):
    q = mp.Queue(); procs = []
    if scheme == "flat":
        procs.append(mp.Process(target=coordinator, args=(2, wan_bytes, rounds, q)))
        procs += [mp.Process(target=worker, args=(COORD[REGION[w]], wan_bytes, wan_bytes, rounds)) for w in range(4)]
    elif scheme == "hier":
        procs.append(mp.Process(target=coordinator, args=(1, wan_bytes, rounds, q)))
        procs += [mp.Process(target=hub, args=(r, wan_bytes, wan_bytes, lan_bytes, rounds, False)) for r in "AB"]
        procs += [mp.Process(target=worker, args=(HUB[REGION[w]], lan_bytes, lan_bytes, rounds)) for w in range(4)]
    else:  # lan only
        hub.q = q
        procs += [mp.Process(target=hub, args=(r, 0, 0, lan_bytes, rounds, True)) for r in "AB"]
        procs += [mp.Process(target=worker, args=(HUB[REGION[w]], lan_bytes, lan_bytes, rounds)) for w in range(4)]
    procs[0].start(); time.sleep(0.3)
    for p in procs[1:]: p.start(); time.sleep(0.05)
    n_results = rounds * (2 if scheme == "lan" else 1)
    res = [q.get(timeout=1800) for _ in range(n_results)]
    for p in procs: p.join(timeout=30)
    if scheme == "lan":   # two hubs report separately; round time = slower region
        res = [max(res[i], res[i + 1]) for i in range(0, len(res), 2)]
    return res


# ---------- network setup ----------
def sh(*cmd, check=True):
    return subprocess.run(list(cmd), check=check, capture_output=True, text=True)

class Net:
    def __init__(self, profile):
        self.p = PROFILES[profile]; self.emu = None
        self.mtu = open("/sys/class/net/lo/mtu").read().strip()
        # Queue on INPUT (receiver side): by then loopback has orphaned the skb from the sending
        # socket, so TCP Small Queues / autocorking on the sender don't see emulated delay as a
        # local queue - matching a real network where packets have left the host.
        self.rule = ["INPUT", "-i", "lo", "-p", "tcp", "-m", "multiport", "--ports", "7001,7002",
                     "-j", "NFQUEUE", "--queue-num", "1", "--queue-bypass"]
    def __enter__(self):
        sh("ip", "link", "set", "lo", "mtu", "1500")            # real-internet packet size
        self.emu = subprocess.Popen([sys.executable, os.path.join(HERE, "wan_emulator.py"), "--ports", "7001,7002",
                                     "--delay-ms", str(self.p["delay_ms"]), "--loss-pct", str(self.p["loss_pct"]),
                                     "--rate-mbit", str(self.p["rate_mbit"])], stdout=subprocess.PIPE, text=True)
        assert "READY" in self.emu.stdout.readline()
        sh("iptables", "-I", *self.rule)
        return self
    def __exit__(self, *exc):
        sh("iptables", "-D", *self.rule, check=False)
        self.emu.terminate(); self.stats = self.emu.communicate(timeout=30)[0].strip()
        sh("ip", "link", "set", "lo", "mtu", self.mtu)


def main():
    global GAP_S
    args = sys.argv[1:]
    b2b = "--b2b" in args            # DDP-style: rounds back to back (gap ~ one compute step)
    args = [a for a in args if a != "--b2b"]
    if b2b:
        GAP_S = 0.05
    N = n_params()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "rounds.csv")
    new = not os.path.exists(path)
    f = open(path, "a", newline=""); w = csv.writer(f)
    if new: w.writerow(["profile", "scheme", "bits", "wan_bytes", "lan_bytes", "round", "round_s"])
    profiles = args or list(PROFILES)
    lan32 = payload_bytes(N, 32)
    print(f"params={N:,}  fp32={lan32:,} B  int8={payload_bytes(N, 8):,} B  int4={payload_bytes(N, 4):,} B", flush=True)
    for prof in profiles:
        rounds = 6 if PROFILES[prof]["rate_mbit"] <= 2 else 10
        with Net(prof) as net:
            combos = [("flat", 32)] if b2b else [(sc, b) for sc in ("flat", "hier") for b in (32, 8, 4)]
            for scheme, bits in combos:
                    wb = payload_bytes(N, bits)
                    res = run_scheme(scheme, wb, lan32, rounds)
                    label = "flat_b2b" if b2b else scheme
                    for i, r in enumerate(res): w.writerow([prof, label, bits, wb, lan32, i, round(r, 4)])
                    f.flush()
                    print(f"{prof:16s} {label:8s} {bits:2d}-bit: median round {sorted(res)[len(res)//2]:.3f} s", flush=True)
        print("   emulator", net.stats, flush=True)
    if ",lan," not in open(path).read():      # local-only regional averaging, measured once
        res = run_scheme("lan", 0, lan32, 10)
        for i, r in enumerate(res): w.writerow(["local", "lan", 32, 0, lan32, i, round(r, 4)])
        print(f"local-only regional averaging: median {sorted(res)[5]:.4f} s", flush=True)
    f.close()


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
