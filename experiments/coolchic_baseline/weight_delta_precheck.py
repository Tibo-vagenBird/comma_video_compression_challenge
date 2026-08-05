#!/usr/bin/env python
"""Go/no-go pre-check for temporal weight delta-coding.

Loads consecutive SAME-ROLE frame_encoder.pt files from a Cool-Chic run's
workdir and, per NN module, compares the exp-Golomb bit cost of:

  * full  : the quantized weights k = round(w / q_step)          (baseline today)
  * delta : k_cur - k_ref, with a SHARED q_step = the reference frame's step
            (the representation the delta-coding design requires)

Same-role = same frame_type AND same display parity (odd<-odd, even<-even) --
i.e. the warm-start chain, which is also the delta reference chain. The I-frame
and the first odd/even P-frame have no same-role predecessor and are skipped
(they'd be coded full anyway).

It mirrors coolchic/bitstream/neuralnet/neuralnet.py::encode_network exactly
(same module/weight-bias traversal, same round(w/q_step)) and uses the real
encode_exp_golomb for costs, so the numbers match what the bitstream would pay.

Run inside the coolchic env:
  python weight_delta_precheck.py --coolchic ~/Cool-Chic \
      --workdir experiments/coolchic_baseline/runs/<run>/work
"""
import argparse
import sys
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

import torch


def _bits(encode_exp_golomb, values, count):
    """Exp-Golomb bit cost of a list of signed ints under a single order."""
    if len(values) == 0:
        return 0
    data_bytes, n_pad = encode_exp_golomb(list(values), [int(count)] * len(values))
    return len(data_bytes) * 8 - int(n_pad)


def _best_delta_bits(encode_exp_golomb, values, possible_counts):
    """Min exp-Golomb cost over the allowed orders -- the delta distribution is
    more peaked at 0 than the full weights, so it gets its own best order."""
    return min(_bits(encode_exp_golomb, values, c) for c in possible_counts)


def _iter_modules(cc_enc, fields_ccc, fields_nn):
    """Mirror encode_network's traversal: yield (module_name, w_or_b, q_step,
    expgol_cnt, possible_counts, param_float_1d) for every populated module."""
    for field_nn in fields_ccc:
        module_name = field_nn.name
        module = getattr(cc_enc, module_name)
        if module is None:
            continue
        for field_wb in fields_nn:
            wb = field_wb.name
            q_step = float(cc_enc.nn_q_step.get_value(module_name, wb))
            expgol_cnt = int(cc_enc.nn_expgol_cnt.get_value(module_name, wb))
            param_dict = module.get_param(which=wb)
            if not param_dict:
                continue
            param = torch.cat([v.flatten() for _, v in param_dict.items()]).detach().cpu()
            yield module_name, wb, q_step, expgol_cnt, param


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coolchic", required=True, type=Path, help="Cool-Chic repo root")
    ap.add_argument("--workdir", required=True, type=Path,
                    help="run .../work dir holding NNNN-frame_encoder.pt files")
    ap.add_argument("--pixels", type=int, default=512 * 384,
                    help="pixels per frame, for bpp (default eval-res 512x384)")
    args = ap.parse_args()

    sys.path.insert(0, str(args.coolchic.expanduser().resolve()))
    from coolchic.component.frame import load_frame_encoder
    from coolchic.bitstream.neuralnet.expgolomb import encode_exp_golomb
    from coolchic.component.core.types import DescriptorCoolChic, DescriptorNN
    from coolchic.nnquant.expgolomb import POSSIBLE_EXP_GOL_COUNT
    fields_ccc = list(fields(DescriptorCoolChic))
    fields_nn = list(fields(DescriptorNN))

    pts = sorted(args.workdir.glob("*-frame_encoder.pt"))
    if not pts:
        sys.exit(f"no *-frame_encoder.pt in {args.workdir}")

    # display order -> loaded frame_encoder (display taken from the filename prefix)
    frames = {}
    for p in pts:
        try:
            display = int(p.name.split("-")[0])
        except ValueError:
            continue
        frames[display] = load_frame_encoder(str(p))
    print(f"loaded {len(frames)} frame encoders from {args.workdir}")

    # group by role = (frame_type, display % 2); same role -> shares a delta chain
    roles = defaultdict(list)
    for display, fe in frames.items():
        roles[(fe.frame_type, display % 2)].append(display)
    role_str = ", ".join(f"{ft}{'odd' if par else 'even'}:{len(v)}"
                         for (ft, par), v in sorted(roles.items()))
    print(f"roles: {role_str}")

    # (cc_name, module, w_or_b) -> accumulators
    agg = defaultdict(lambda: {"n": 0, "full": 0, "delta": 0, "match": 0, "pairs": 0})
    n_pairs = 0
    skipped = 0

    for (ft, par), displays in sorted(roles.items()):
        displays = sorted(displays)
        for ref_d, cur_d in zip(displays[:-1], displays[1:]):
            fe_ref, fe_cur = frames[ref_d], frames[cur_d]
            common = set(fe_ref.coolchic_enc) & set(fe_cur.coolchic_enc)
            n_pairs += 1
            for cc_name in sorted(common):
                ref_mods = {(m, wb): (q, c, p) for m, wb, q, c, p in
                            _iter_modules(fe_ref.coolchic_enc[cc_name], fields_ccc, fields_nn)}
                for m, wb, q_cur, cnt_cur, p_cur in _iter_modules(
                        fe_cur.coolchic_enc[cc_name], fields_ccc, fields_nn):
                    ref = ref_mods.get((m, wb))
                    if ref is None:
                        continue
                    q_ref, _cnt_ref, p_ref = ref
                    if p_cur.numel() != p_ref.numel():
                        skipped += 1
                        continue
                    poss = POSSIBLE_EXP_GOL_COUNT.get_value(m, wb)

                    k_full = torch.round(p_cur / q_cur).to(torch.int32)
                    k_ref = torch.round(p_ref / q_ref).to(torch.int32)
                    # force shared q_step = reference's, as the design requires
                    k_cur_shared = torch.round(p_cur / q_ref).to(torch.int32)
                    delta = (k_cur_shared - k_ref).tolist()

                    a = agg[(cc_name, m, wb)]
                    a["n"] += p_cur.numel()
                    a["full"] += _bits(encode_exp_golomb, k_full.tolist(), cnt_cur)
                    a["delta"] += _best_delta_bits(encode_exp_golomb, delta, poss)
                    a["match"] += int(q_cur == q_ref)
                    a["pairs"] += 1

    if n_pairs == 0:
        sys.exit("no same-role consecutive pairs found (need >=2 frames per role)")

    # -------- report --------
    print(f"\nsame-role consecutive pairs: {n_pairs}"
          + (f"  (skipped {skipped} arch-mismatched module instances)" if skipped else ""))
    print(f"\n{'cc':8} {'module':11} {'w/b':6} {'n_param':>8} "
          f"{'full_bits':>10} {'delta_bits':>10} {'saving%':>8} {'qstep_match':>11}")
    tot_full = tot_delta = tot_n = 0
    tot_match = tot_slots = 0
    for key in sorted(agg):
        cc_name, m, wb = key
        a = agg[key]
        save = 100.0 * (1 - a["delta"] / a["full"]) if a["full"] else 0.0
        match = 100.0 * a["match"] / a["pairs"] if a["pairs"] else 0.0
        print(f"{cc_name:8} {m:11} {wb:6} {a['n']:>8} "
              f"{a['full']:>10} {a['delta']:>10} {save:>7.1f}% {match:>10.0f}%")
        tot_full += a["full"]; tot_delta += a["delta"]; tot_n += a["n"]
        tot_match += a["match"]; tot_slots += a["pairs"]

    tot_save = 100.0 * (1 - tot_delta / tot_full) if tot_full else 0.0
    tot_match_pct = 100.0 * tot_match / tot_slots if tot_slots else 0.0
    print(f"{'TOTAL':8} {'':11} {'':6} {tot_n:>8} "
          f"{tot_full:>10} {tot_delta:>10} {tot_save:>7.1f}%")

    # per-frame nn rate (bits summed over pairs -> per-pair average -> bpp)
    full_bpp = tot_full / (n_pairs * args.pixels)
    delta_bpp = tot_delta / (n_pairs * args.pixels)
    print(f"\nper-frame nn rate:  full = {full_bpp:.6f} bpp   "
          f"delta = {delta_bpp:.6f} bpp   ({-tot_save:+.1f}%)")
    print(f"q_step consistency: {tot_match_pct:.1f}% of module-steps already "
          f"match the reference (higher = forcing a shared step costs less)")

    verdict = ("GREEN  -- build it" if tot_save >= 40 else
               "YELLOW -- marginal, weigh against the bitstream complexity"
               if tot_save >= 15 else "RED    -- not worth it")
    print(f"\nVERDICT: {verdict}  (delta saves {tot_save:.1f}% of the NN-weight rate)")
    print("note: this is the NN-weight portion only; compare against the run's "
          "total rate (nn_bpp vs latent_bpp) to gauge the score impact.")


if __name__ == "__main__":
    main()
