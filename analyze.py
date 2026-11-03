"""Combines the convergence grid (results/sim), measured sync rounds (results/comm) and the
end-to-end check (results/dist) into tables and charts.

Wall-clock model per config c and network profile p (2,000 steps per worker):
    T = steps * t_compute  +  WAN_syncs(c) * round_wan(p, scheme, bits)  +  LAN_only_syncs(c) * round_lan
t_compute comes from the real distributed run (per-worker step time with 4 workers in parallel).
DDP uses the back-to-back ("flat_b2b") round time, because it syncs after every step.
"""
import glob, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"; OUT = f"{R}/charts"; os.makedirs(OUT, exist_ok=True)
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0"
FAM = {"ddp": "#8a8984", "diloco": "#2a78d6", "hier": "#eb6834"}
MARK = {32: "o", 8: "s", 4: "^"}
PROFILES = ["regional", "intercontinental", "constrained", "poor"]
PLABEL = {"regional": "Regional\n100 Mbit/s, 20 ms RTT", "intercontinental": "Intercontinental\n50 Mbit/s, 150 ms RTT, 0.1% loss",
          "constrained": "Constrained\n10 Mbit/s, 200 ms RTT, 0.5% loss", "poor": "Poor\n2 Mbit/s, 300 ms RTT, 1% loss"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#8a8984", "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})


def label(r):
    if r["method"] == "ddp": return "DDP (sync every step)"
    b = "" if r["bits"] == 32 else f", INT{r['bits']}"
    if r["method"] == "diloco": return f"DiLoCo H={r['H']}{b}"
    return f"Hierarchical H={r['H_r']}/{r['H_g']}{b}"


def load_sim():
    rows, curves = [], []
    for f in glob.glob(f"{R}/sim/*_s[12].json"):
        r = json.load(open(f))
        rows.append({"name": r["name"], "method": r["method"], "H": r["H"], "H_r": r["H_r"], "H_g": r["H_g"],
                     "bits": r["bits"] if r["method"] != "ddp" else 32, "seed": r["seed"], "final_val": r["final_val"],
                     "wan_syncs": r["wan_syncs"], "lan_syncs": r["lan_syncs"], "steps": r["steps"]})
        for h in r["history"]:
            curves.append({"name": r["name"], "seed": r["seed"], "step": h["step"], "val": h["val_loss"]})
    return pd.DataFrame(rows), pd.DataFrame(curves)


def main():
    sim, curves = load_sim()
    comm = pd.read_csv(f"{R}/comm/rounds.csv")
    rt = comm.groupby(["profile", "scheme", "bits"]).round_s.median()
    lan = rt[("local", "lan", 32)]
    dist = [json.load(open(f)) for f in glob.glob(f"{R}/dist/*/summary.json")]
    t_c = float(np.mean([d["t_compute_per_step"] for d in dist]))

    agg = sim.groupby(["name", "method", "H", "H_r", "H_g", "bits", "wan_syncs", "lan_syncs", "steps"]).final_val \
             .agg(["mean", "min", "max", "count"]).reset_index()
    rows = []
    for r in agg.to_dict("records"):
        for p in PROFILES:
            if r["method"] == "ddp":
                wan = rt[(p, "flat_b2b", 32)]; lan_only = 0
            elif r["method"] == "diloco":
                wan = rt[(p, "flat", r["bits"])]; lan_only = 0
            else:
                wan = rt[(p, "hier", r["bits"])]; lan_only = r["lan_syncs"] - r["wan_syncs"]
            comm_s = r["wan_syncs"] * wan + lan_only * lan
            rows.append({**r, "label": label(r), "profile": p, "compute_s": r["steps"] * t_c, "comm_s": comm_s,
                         "total_s": r["steps"] * t_c + comm_s, "round_s": wan})
    T = pd.DataFrame(rows)
    T.to_csv(f"{R}/wallclock.csv", index=False)

    # ---- Fig 1: convergence (fp32 variants) ----
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    show = [("ddp", "DDP (sync every step)", FAM["ddp"], "-"), ("diloco_H10_b32", "DiLoCo H=10", "#86b6ef", "-"),
            ("diloco_H50_b32", "DiLoCo H=50", FAM["diloco"], "-"), ("diloco_H100_b32", "DiLoCo H=100", "#1c5cab", "-"),
            ("hier_Hr10_Hg50_b32", "Hierarchical H=10/50", FAM["hier"], "-"), ("hier_Hr10_Hg100_b32", "Hierarchical H=10/100", "#b04a1c", "--")]
    for name, lab, col, ls in show:
        c = curves[curves.name == name].groupby("step").val.mean()
        if len(c): ax.plot(c.index, c.values, color=col, lw=2, ls=ls, label=lab)
    ax.set_xlabel("training steps per worker"); ax.set_ylabel("validation loss (lower is better)")
    ax.set_ylim(1.55, 2.6); ax.grid(color=GRID); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Learning curves (full precision, mean of 2 seeds)", loc="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_convergence.png", dpi=200); plt.close(fig)

    # ---- Fig 2: effect of compression on final loss ----
    fig, ax = plt.subplots(figsize=(7.5, 4))
    groups = [("diloco", 50, None), ("diloco", 100, None), ("hier", None, 50), ("hier", None, 100)]
    names = ["DiLoCo H=50", "DiLoCo H=100", "Hier. H=10/50", "Hier. H=10/100"]
    for gi, (m, H, Hg) in enumerate(groups):
        for bi, b in enumerate([32, 8, 4]):
            q = agg[(agg.method == m) & (agg.bits == b) & ((agg.H == H) if H else (agg.H_g == Hg))]
            if not len(q): continue
            q = q.iloc[0]; x = gi + (bi - 1) * 0.22
            ax.bar(x, q["mean"] - 1.5, 0.2, bottom=1.5, color=["#1c5cab", "#3987e5", "#9ec5f4"][bi], label=f"{'FP32' if b == 32 else 'INT' + str(b)}" if gi == 0 else None)
            ax.errorbar(x, q["mean"], yerr=[[q["mean"] - q["min"]], [q["max"] - q["mean"]]], fmt="none", ecolor=INK2, capsize=2)
            ax.text(x, q["max"] + 0.004, f"{q['mean']:.3f}", ha="center", fontsize=6.5, color=INK2)
    d = agg[agg.method == "ddp"]["mean"].iloc[0]
    ax.axhline(d, color=FAM["ddp"], ls="--", lw=1); ax.text(3.45, d + 0.004, f"DDP {d:.3f}", ha="right", fontsize=7.5, color=INK2)
    ax.set_xticks(range(4), names); ax.set_ylim(1.5, None); ax.set_ylabel("final validation loss")
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True); ax.legend(frameon=False, ncols=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_title("Does compressing the updates hurt the model?", loc="left", fontsize=11, color=INK, pad=24)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_compression_quality.png", dpi=200); plt.close(fig)

    # ---- Fig 3: measured sync round times ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, sc, title in [(axes[0], "flat", "Flat (every worker crosses the WAN)"), (axes[1], "hier", "Hierarchical (one update per region)")]:
        for bi, b in enumerate([32, 8, 4]):
            vals = [rt.get((p, sc, b), np.nan) for p in PROFILES]
            xs = np.arange(4) + (bi - 1) * 0.27
            bars = ax.bar(xs, vals, 0.25, color=["#1c5cab", "#3987e5", "#9ec5f4"][bi], label="FP32" if b == 32 else f"INT{b}")
            for bb, v in zip(bars, vals):
                ax.text(bb.get_x() + bb.get_width() / 2, v * 1.08, f"{v:.2f}", ha="center", fontsize=6.5, color=INK2, rotation=90)
        ax.set_yscale("log"); ax.set_xticks(range(4), [p.capitalize() for p in PROFILES])
        ax.grid(axis="y", color=GRID); ax.set_axisbelow(True); ax.set_title(title, loc="left", fontsize=10, color=INK)
    axes[0].set_ylabel("seconds per sync (log scale)"); axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Measured time for one synchronisation over the emulated WAN (median)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_sync_times.png", dpi=200); plt.close(fig)

    # ---- Fig 4: time vs quality, one panel per profile ----
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, p in zip(axes.flat, PROFILES):
        q = T[T.profile == p]
        for r in q.to_dict("records"):
            ax.scatter(r["total_s"] / 60, r["mean"], color=FAM[r["method"]], marker=MARK[r["bits"]], s=45,
                       edgecolor=SURF, linewidth=1.5, zorder=3)
        ax.set_xscale("log"); ax.grid(color=GRID); ax.set_axisbelow(True)
        ax.set_title(PLABEL[p], loc="left", fontsize=9, color=INK)
        ax.set_xlabel("estimated training time (minutes, log scale)"); ax.set_ylabel("final validation loss")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=FAM[k], marker="o", ls="", label=l) for k, l in [("ddp", "DDP"), ("diloco", "DiLoCo"), ("hier", "Hierarchical")]]
    h += [Line2D([], [], color=INK2, marker=MARK[b], ls="", label=("FP32" if b == 32 else f"INT{b}")) for b in (32, 8, 4)]
    fig.legend(handles=h, frameon=False, ncols=6, loc="upper left", bbox_to_anchor=(0.01, 0.95))
    fig.suptitle("Training time vs model quality (lower-left is better)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(f"{OUT}/fig4_time_vs_quality.png", dpi=200); plt.close(fig)

    # ---- tables ----
    piv = T.pivot_table(index="label", columns="profile", values="total_s")[PROFILES] / 60
    piv["final_val"] = T.groupby("label")["mean"].first()
    piv = piv.sort_values("final_val")
    with open(f"{R}/key_numbers.md", "w") as f:
        f.write(f"t_compute per step (real 4-process run): {t_c * 1000:.1f} ms\n\n")
        f.write("## Estimated minutes for 2,000 steps + final val loss (mean of seeds 1-2)\n\n")
        f.write(piv.round(2).to_markdown() + "\n\n")
        f.write("## Median sync round (s)\n\n" + rt.unstack("bits").round(3).to_markdown() + "\n\n")
        f.write("## Final val loss per config (seeds)\n\n" + agg.round(4).to_markdown(index=False) + "\n\n")
        if dist:
            f.write("## End-to-end distributed runs\n\n")
            for d in dist:
                f.write(f"- {d['profile']} {d['bits']}-bit: final val {d['final_val']:.4f}; rounds {np.round(d['round_times'], 3).tolist()}; "
                        f"t_compute/step {d['t_compute_per_step'] * 1000:.1f} ms; total {d['t_total']:.1f} s\n")
    print(open(f"{R}/key_numbers.md").read())


if __name__ == "__main__":
    main()
