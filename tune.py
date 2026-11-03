"""Small hyperparameter sweep on seed 0 (final results use seeds 1-2).
DDP: inner learning rate. DiLoCo (H=50): inner lr x outer lr. Writes results/sim/best_hparams.json."""
import json, os, time
import core

RUNS = [("ddp", dict(lr=1e-3)), ("ddp", dict(lr=2e-3)), ("ddp", dict(lr=4e-3)),
        ("diloco", dict(H=50, lr=2e-3, outer_lr=0.5)), ("diloco", dict(H=50, lr=2e-3, outer_lr=0.7)),
        ("diloco", dict(H=50, lr=2e-3, outer_lr=1.0)), ("diloco", dict(H=50, lr=4e-3, outer_lr=0.7)),
        # extension: the first sweep's winners sat at the edge of the range
        ("ddp", dict(lr=8e-3)), ("diloco", dict(H=50, lr=4e-3, outer_lr=1.0)),
        ("diloco", dict(H=50, lr=8e-3, outer_lr=0.7)), ("diloco", dict(H=50, lr=8e-3, outer_lr=1.0)),
        # second extension: DDP's best was again at the top of the range (lr 8e-3)
        ("ddp", dict(lr=1.6e-2)), ("diloco", dict(H=50, lr=1.6e-2, outer_lr=0.7)), ("diloco", dict(H=50, lr=1.6e-2, outer_lr=1.0))]
def main():
    os.makedirs("results/tune", exist_ok=True)
    res = []
    for method, kw in RUNS:
        tag = method + "_" + "_".join(f"{k}{v}" for k, v in kw.items())
        path = f"results/tune/{tag}.json"
        if not os.path.exists(path):
            t0 = time.time()
            r = core.run(method, steps=2000, eval_every=500, seed=0, log=lambda s: None, **kw)
            r["kw"] = kw; r["wall_s"] = time.time() - t0
            json.dump(r, open(path, "w"))
            print(f"{tag:40s} final val {r['final_val']:.4f}", flush=True)
        res.append(json.load(open(path)))
    best = {}
    for m in ("ddp", "diloco"):
        b = min((r for r in res if r["method"] == m), key=lambda r: r["final_val"])
        best[m] = b["kw"]; print("best", m, b["kw"], round(b["final_val"], 4))
    json.dump(best, open("results/sim/best_hparams.json", "w"), indent=1)


if __name__ == "__main__":
    main()
