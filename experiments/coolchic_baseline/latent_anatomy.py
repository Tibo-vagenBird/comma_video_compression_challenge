#!/usr/bin/env python
"""Per-latent-grid bit allocation for a Cool-Chic run.

Answers: does the multi-resolution latent hierarchy spend bits on the fine
(high-frequency) levels, or does rate-aware training self-prune them? For a
metric that only needs low/mid frequencies, we expect the fine grids to carry
~0 bits if the hierarchy is behaving.

Run in the *coolchic* env, pointed at a run's workdir:
  python latent_anatomy.py --coolchic ~/comma_challenge/Cool-Chic \
      --run experiments/coolchic_baseline/runs/lmbda_40_n32_ippp_eval_comma_r01

Loads each XXXX-frame_encoder.pt, computes exact per-grid latent rate (bits) via
the ARM, and reports coarse->fine bit distribution, aggregated over frame roles
(intra / judged-odd / carrier-even).
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch


def grid_rates(cc_enc):
    """Return list of (h, w, n_elem, bits) per latent grid for one CoolChicEncoder."""
    dec_lat = cc_enc.get_quantize_latent(
        quantizer_noise_type="none", quantizer_type="hardround",
        soft_round_temperature=torch.tensor(1e-4),
        noise_parameter=torch.tensor(1.0), AC_MAX_VAL=-1,
    )
    flat_rate, _, _ = cc_enc.get_rate_latent(dec_lat)  # bits per element, grid order
    out, off = [], 0
    for g in dec_lat:
        n = g.numel()
        h, w = g.shape[-2], g.shape[-1]
        out.append((h, w, n, flat_rate[off:off + n].sum().item()))
        off += n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coolchic", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path)
    args = ap.parse_args()
    sys.path.insert(0, str(args.coolchic.expanduser().resolve()))
    from coolchic.component.frame import load_frame_encoder

    work = (args.run / "work").resolve()
    pts = sorted(work.glob("*-frame_encoder.pt"))
    if not pts:
        raise SystemExit(f"no *-frame_encoder.pt in {work}")

    # role buckets; per-grid-index accumulators of bits and a resolution label
    roles = defaultdict(lambda: defaultdict(lambda: {"res": None, "cc": defaultdict(float)}))
    role_bytes = defaultdict(float)
    role_count = defaultdict(int)

    for pt in pts:
        disp = int(pt.name.split("-")[0])
        role = "intra" if disp == 0 else ("judged-odd" if disp % 2 == 1 else "carrier-even")
        fe = load_frame_encoder(str(pt))
        role_count[role] += 1
        for cc_name, cc_enc in fe.coolchic_enc.items():  # 'residue', maybe 'motion'
            try:
                gr = grid_rates(cc_enc)
            except Exception as e:  # noqa: BLE001
                print(f"  ! {pt.name} [{cc_name}] skipped: {e}", flush=True)
                continue
            for gi, (h, w, n, bits) in enumerate(gr):
                slot = roles[role][gi]
                slot["res"] = f"{h}x{w}"
                slot["cc"][cc_name] += bits
                role_bytes[role] += bits / 8

    print(f"run: {args.run}")
    for role in ("intra", "judged-odd", "carrier-even"):
        if role not in roles:
            continue
        n = role_count[role]
        tot = role_bytes[role]
        print(f"\n=== {role}  ({n} frames, {tot/n:,.0f} latent B/frame avg) ===")
        print(f"  {'grid':>4} {'res':>9} {'bits/frame':>11} {'% of latent':>11}")
        role_total_bits = sum(sum(s['cc'].values()) for s in roles[role].values())
        for gi in sorted(roles[role]):
            s = roles[role][gi]
            b = sum(s["cc"].values())
            pct = 100 * b / role_total_bits if role_total_bits else 0
            print(f"  {gi:>4} {s['res']:>9} {b/n:>11.1f} {pct:>10.1f}%")


if __name__ == "__main__":
    main()
