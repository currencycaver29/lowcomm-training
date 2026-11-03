"""End-to-end validation: real distributed DiLoCo vs. simulation, and measured vs. benchmarked sync time.
Runs (root, nothing else running): python validate.py   -> results/dist/<cfg>/summary.json, results/validation.json"""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
best = json.load(open(f"{HERE}/results/sim/best_hparams.json"))["diloco"]
CFGS = [dict(bits=32), dict(bits=8)]
STEPS, H, SEED, PROFILE = 300, 50, 1, "constrained"
out = {}
for c in CFGS:
    tag = f"{PROFILE}_b{c['bits']}"
    d = f"{HERE}/results/dist/{tag}"
    if not os.path.exists(f"{d}/summary.json"):
        subprocess.run([sys.executable, "dist_train.py", "--profile", PROFILE, "--bits", str(c["bits"]), "--steps", str(STEPS),
                        "--H", str(H), "--seed", str(SEED), "--lr", str(best["lr"]), "--outer-lr", str(best["outer_lr"]),
                        "--out", d], cwd=HERE, check=True)
    dist = json.load(open(f"{d}/summary.json"))
    sim_path = f"{d}/sim_twin.json"
    if not os.path.exists(sim_path):
        code = (f"import json, core; r = core.run('diloco', steps={STEPS}, H={H}, bits={c['bits']}, seed={SEED}, lr={best['lr']}, "
                f"outer_lr={best['outer_lr']}, eval_every={STEPS}, log=lambda s: None); json.dump(r, open('{sim_path}', 'w'))")
        subprocess.run([sys.executable, "-c", code], cwd=HERE, check=True)
    sim = json.load(open(sim_path))
    out[tag] = {"dist_final_val": dist["final_val"], "sim_final_val": sim["final_val"],
                "dist_round_times": dist["round_times"], "t_compute_per_step": dist["t_compute_per_step"], "t_total": dist["t_total"]}
    print(tag, json.dumps(out[tag]), flush=True)
json.dump(out, open(f"{HERE}/results/validation.json", "w"), indent=1)
