"""Runs the convergence grid (simulated workers, exact math). Resumable: skips finished runs.
Usage: python run_grid.py [seed ...]     -> results/sim/<name>_s<seed>.json
"""
import json, os, sys, time
import core

STEPS = 2000
GRID = [("ddp", dict())]
GRID += [("diloco", dict(H=10, bits=32))]
GRID += [("diloco", dict(H=H, bits=b)) for H in (50, 100) for b in (32, 8, 4)]
GRID += [("hier", dict(H_r=10, H_g=Hg, bits=b)) for Hg in (50, 100) for b in (32, 8, 4)]


def name(method, kw):
    if method == "ddp": return "ddp"
    if method == "diloco": return f"diloco_H{kw['H']}_b{kw['bits']}"
    return f"hier_Hr{kw['H_r']}_Hg{kw['H_g']}_b{kw['bits']}"


if __name__ == "__main__":
    seeds = [int(x) for x in sys.argv[1:]] or [1, 2]
    best = json.load(open("results/sim/best_hparams.json"))   # from tune.py (seed 0)
    hp = {"ddp": {"lr": best["ddp"]["lr"]},
          "diloco": {k: best["diloco"][k] for k in ("lr", "outer_lr")},
          "hier": {k: best["diloco"][k] for k in ("lr", "outer_lr")}}   # hierarchical reuses DiLoCo's tuned values
    os.makedirs("results/sim", exist_ok=True)
    for seed in seeds:
        for method, kw in GRID:
            out = f"results/sim/{name(method, kw)}_s{seed}.json"
            if os.path.exists(out) and json.load(open(out)).get("hparams") == hp[method]:
                continue                     # done with the current tuned settings
            t0 = time.time()
            r = core.run(method, steps=STEPS, eval_every=100, seed=seed, log=lambda s: None, **kw, **hp[method])
            r["hparams"] = hp[method]
            r["wall_s"] = time.time() - t0; r["name"] = name(method, kw)
            json.dump(r, open(out, "w"))
            print(f"{r['name']:24s} seed {seed}: final val {r['final_val']:.4f}  ({r['wall_s']:.0f} s)", flush=True)
