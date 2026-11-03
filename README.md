# Low-communication training over bad networks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23889666.svg)](https://doi.org/10.5281/zenodo.23889666)

**Paper:** *Does Low-Communication Training Survive Bad Networks? A Reproducible Study of DDP, DiLoCo and Hierarchical Aggregation over Emulated WAN Links*. Divij Karanam, 2026. Preprint on Zenodo: https://doi.org/10.5281/zenodo.23889666 ([PDF in this repo](paper/main.pdf))

<details><summary>Cite</summary>

```bibtex
@misc{karanam2026lowcomm,
  author    = {Karanam, Divij},
  title     = {Does Low-Communication Training Survive Bad Networks? A Reproducible Study of DDP, DiLoCo and Hierarchical Aggregation over Emulated WAN Links},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23889666},
  url       = {https://doi.org/10.5281/zenodo.23889666}
}
```
</details>

Reproducible CPU-scale study comparing **DDP**, **DiLoCo** and **two-level hierarchical DiLoCo**
(with FP32 / INT8 / INT4 compression of WAN traffic) for training a small transformer language model
with 4 workers in 2 emulated regions, under 4 emulated WAN profiles.

## Files

| File | What it does |
|---|---|
| `core.py` | Model (GPT-style char LM, 351,744 params), data, the three sync strategies, blockwise INT8/INT4 quantization |
| `tune.py` | Hyperparameter sweep on seed 0 → `results/sim/best_hparams.json` |
| `run_grid.py` | 14 configurations × seeds 1–2 → `results/sim/*.json` (resumable) |
| `wan_emulator.py` | Userspace WAN emulator on Linux NFQUEUE: per-link full-duplex queues, rate, delay, loss, BDP drop-tail buffer |
| `comm_bench.py` | Measures real TCP synchronization rounds (flat / hierarchical / local) over the emulated WAN → `results/comm/rounds.csv`; `--b2b` = DDP-style back-to-back rounds |
| `dist_train.py`, `validate.py` | Real 4-process distributed DiLoCo with real quantized payloads, compared with the simulation → `results/validation.json` |
| `analyze.py` | Wall-clock model, tables and charts → `results/key_numbers.md`, `results/wallclock.csv`, `results/charts/` |

## Reproduce

Linux, Python 3.11+, root (for iptables / NFQUEUE / loopback MTU).
```
pip install "jax[cpu]" optax numpy pandas matplotlib tabulate NetfilterQueue   # NetfilterQueue needs libnetfilter-queue-dev
python3 tune.py
python3 run_grid.py 1 2
sudo python3 comm_bench.py
sudo python3 comm_bench.py --b2b
sudo python3 validate.py
python3 analyze.py
```
Run the network steps (`comm_bench.py`, `validate.py`) on an otherwise idle machine, and never at the same time as training.

## Method notes

- **Emulator on the receive path.** Queueing on loopback OUTPUT left delayed packets charged to the sending socket, and Linux TCP Small Queues then throttled a 100 Mbit/s link to about 4 Mbit/s. The emulator therefore hooks INPUT, after loopback has orphaned the packet.
- **MTU 1500** is set on loopback during measurements (the default 64 kB makes TCP far too tolerant of loss), then restored.
- **Downlink also quantized:** the averaged update is sent back in the same INT8/INT4 format, in both the simulation and the benchmark.
- **Tuning:** the sweep was extended twice because the best learning rates initially sat at the edge of the range. Final: DDP lr 0.016; DiLoCo inner lr 0.016, outer lr 0.7 (Nesterov momentum 0.9). Hierarchical reuses DiLoCo's values.
- **Wall-clock model:** T = steps × t_compute + WAN syncs × WAN round + LAN-only syncs × LAN round. Validated against real distributed runs: it underestimated total time by 8% (FP32) and 15% (INT8), mostly because of worker skew and payload encoding, so treat estimates as lower bounds.
- `results/sim/superseded/` holds 5 grid runs made with pre-extension hyperparameters (not used in the paper).

## Environment of the published run

4 vCPU Intel Xeon @ 2.80 GHz, Linux 6.18 (no netem), Python 3.13, JAX 0.12, optax 0.2.8, NetfilterQueue 1.1.0.
