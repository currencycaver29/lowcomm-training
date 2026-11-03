"""Full-duplex, per-link WAN emulator on Linux NFQUEUE (for kernels without netem).

Each region's WAN link is identified by the coordinator port it connects to
(e.g. 7001 = region A, 7002 = region B). Every link has two independent
directions (up: dst port == link port, down: src port == link port), each a
FIFO with: serialisation at `rate`, then propagation `delay`, random `loss`,
and a drop-tail buffer of one bandwidth-delay product (min 30 packets).

Attach it to the INPUT hook (iptables -I INPUT -i lo ... -j NFQUEUE --queue-num 1): queueing on
OUTPUT would leave the packets charged to the sending socket, so Linux TCP Small Queues throttles
the sender (measured: 4.4 Mbit/s on a 100 Mbit link), an artefact a real network doesn't have.

Usage (root): python wan_emulator.py --ports 7001,7002 --delay-ms 75 --loss-pct 0.1 --rate-mbit 50
"""
import argparse, heapq, random, select, signal, time
from netfilterqueue import NetfilterQueue

ap = argparse.ArgumentParser()
ap.add_argument("--ports", required=True)
ap.add_argument("--delay-ms", type=float, required=True)
ap.add_argument("--loss-pct", type=float, default=0.0)
ap.add_argument("--rate-mbit", type=float, required=True)
ap.add_argument("--queue-num", type=int, default=1)
a = ap.parse_args()

PORTS = {int(p) for p in a.ports.split(",")}
DELAY, LOSS, RATE = a.delay_ms / 1000, a.loss_pct / 100, a.rate_mbit * 1e6
BUF_BYTES = max(RATE / 8 * 2 * DELAY, 30 * 1500)          # one BDP (round trip), at least 30 packets
state = {}                                                  # (port, dir) -> {"free": t, "queued": bytes timeline}
heap, seq = [], 0
stats = {"pkts": 0, "loss_drops": 0, "buf_drops": 0}


def classify(b):
    ihl = (b[0] & 0x0F) * 4
    sport = int.from_bytes(b[ihl:ihl + 2], "big"); dport = int.from_bytes(b[ihl + 2:ihl + 4], "big")
    if dport in PORTS: return (dport, "up")
    if sport in PORTS: return (sport, "down")
    return None


def on_packet(pkt):
    global seq
    b = pkt.get_payload()
    key = classify(b)
    if key is None:
        pkt.accept(); return
    stats["pkts"] += 1
    now = time.monotonic()
    st = state.setdefault(key, {"free": 0.0})
    backlog = max(0.0, st["free"] - now) * RATE / 8        # bytes waiting to be serialised
    if backlog + len(b) > BUF_BYTES:
        stats["buf_drops"] += 1; pkt.drop(); return
    if LOSS and random.random() < LOSS:
        stats["loss_drops"] += 1; pkt.drop(); return
    start = max(now, st["free"])
    st["free"] = start + len(b) * 8 / RATE
    release = st["free"] + DELAY
    pkt.retain()
    heapq.heappush(heap, (release, seq, pkt)); seq += 1


nfq = NetfilterQueue()
nfq.bind(a.queue_num, on_packet, max_len=50000)
fd = nfq.get_fd()
running = [True]
signal.signal(signal.SIGTERM, lambda *x: running.__setitem__(0, False))
print("EMULATOR READY", flush=True)
try:
    while running[0]:
        timeout = max(0.0, heap[0][0] - time.monotonic()) if heap else 0.2
        if select.select([fd], [], [], timeout)[0]:
            nfq.run(block=False)
        now = time.monotonic()
        while heap and heap[0][0] <= now:
            heapq.heappop(heap)[2].accept()
finally:
    for _, _, p in heap: p.accept()
    nfq.unbind()
    print(f"STATS {stats}", flush=True)
