#!/usr/bin/env python
"""End-to-end byte-exact gate for delta coding and incremental reconstruction.

Deterministic: it does NOT train. It takes a FINISHED run's workdir (the
NNNN-frame_encoder.pt files) and, from the SAME weights, encodes the bitstream
twice --

    * delta OFF  -> every module coded full   (baseline format)
    * delta ON   -> same-role predecessor delta where cheaper

-- then decodes both and asserts that full, delta, and their encoder-side
incremental reconstructions are byte-identical while the delta bitstream is no
larger. Any difference means the fast reference path or delta coding is invalid.

Run in the coolchic env:
  python experiments/coolchic_baseline/test_delta_e2e.py \
      --coolchic ~/Cool-Chic \
      --workdir experiments/coolchic_baseline/runs/<run>/work \
      --input   experiments/coolchic_baseline/data/video0_512x384_20p_yuv420_8b.yuv \
      --n_frames 6 --p_pos 1-5 --intra_pos 0
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coolchic", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--n_frames", type=int, required=True)
    ap.add_argument("--p_pos", default="")
    ap.add_argument("--intra_pos", default="0")
    ap.add_argument("--frame_offset", type=int, default=0)
    ap.add_argument("--no-decode", action="store_true",
                    help="skip the (slow) byte-exact decode check; just report the "
                         "full-vs-delta saving. Use once correctness is already proven.")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(os.path.expanduser(args.coolchic)))
    import torch  # noqa: F401
    from coolchic.bitstream.decode import decode_video
    from coolchic.bitstream.encode import encode_frame_with_reconstruction
    from coolchic.bitstream.neuralnet.delta import find_same_role_reference
    from coolchic.component.frame import load_frame_encoder
    from coolchic.component.video import _get_frame_path_prefix
    from coolchic.utils.codingstructure import CodingStructure
    from coolchic.utils.parsecli import get_coding_structure_from_args

    os.chdir(args.workdir)  # frame_encoder.pt files & decoded refs live here (relative)

    # argparse.Namespace (not SimpleNamespace) -- get_coding_structure_from_args
    # uses `"input" in args`, which needs Namespace's __contains__.
    cs_args = argparse.Namespace(
        input=args.input, intra_pos=args.intra_pos, p_pos=args.p_pos,
        n_frames=args.n_frames, frame_offset=args.frame_offset,
    )
    coding_structure = CodingStructure(**get_coding_structure_from_args(cs_args))

    def load_fe(display):
        return load_frame_encoder(f"{_get_frame_path_prefix(display)}frame_encoder.pt")

    def encode_all(out_path, delta):
        incremental_recon = {}
        for coding_idx in range(args.n_frames):
            frame = coding_structure.get_frame_from_coding_order(coding_idx)
            fe = load_fe(frame.display_order)
            refs_data = [
                incremental_recon[idx_ref]
                for idx_ref in frame.index_references
            ]
            ref_fe = None
            if delta:
                ref = find_same_role_reference(coding_structure, frame)
                if ref is not None:
                    ref_fe = load_fe(ref.display_order)
            _, reconstructed = encode_frame_with_reconstruction(
                fe,
                out_path,
                coding_structure,
                reference_frames=refs_data,
                ref_frame_encoder=ref_fe,
            )
            incremental_recon[frame.display_order] = reconstructed
        return incremental_recon

    full_bs = os.path.abspath("_delta_e2e_full.cool")
    delta_bs = os.path.abspath("_delta_e2e_delta.cool")
    print("encoding delta OFF (full) ...", flush=True)
    incremental_full = encode_all(full_bs, delta=False)
    print("encoding delta ON  ...", flush=True)
    incremental_delta = encode_all(delta_bs, delta=True)

    n_full = os.path.getsize(full_bs)
    n_delta = os.path.getsize(delta_bs)
    print(f"\nbitstream bytes: full={n_full:,}  delta={n_delta:,}  "
          f"saving={100 * (1 - n_delta / n_full):+.1f}%")

    if args.no_decode:
        for p in (full_bs, delta_bs):
            try:
                os.remove(p)
            except OSError:
                pass
        print("(--no-decode: skipped byte-exact check; correctness proven separately)")
        sys.exit(0 if n_delta <= n_full else 1)

    print("decoding both ...", flush=True)
    dec_full = decode_video(full_bs, decoded_path=None, max_decoding_order=args.n_frames - 1)
    dec_delta = decode_video(delta_bs, decoded_path=None, max_decoding_order=args.n_frames - 1)

    # ---- byte-exact comparison of every decoded and incremental frame ----
    ok = True
    for k in dec_full:
        display_idx = int(k)
        comparisons = {
            "full-delta": (dec_full[k].data, dec_delta[k].data),
            "incremental-full": (
                incremental_full[display_idx].data,
                dec_full[k].data,
            ),
            "incremental-delta": (
                incremental_delta[display_idx].data,
                dec_delta[k].data,
            ),
        }
        max_diffs = {}
        for name, (a, b) in comparisons.items():
            if isinstance(a, dict):  # yuv420
                md = max((a[c] - b[c]).abs().max().item() for c in a)
            else:
                md = (a - b).abs().max().item()
            max_diffs[name] = md
        if any(md != 0 for md in max_diffs.values()):
            ok = False
        print(
            f"  frame {k:>3}: "
            + "  ".join(f"{name}={md}" for name, md in max_diffs.items())
            + ("   OK" if all(md == 0 for md in max_diffs.values()) else "   DIFF")
        )

    for p in (full_bs, delta_bs):
        try:
            os.remove(p)
        except OSError:
            pass

    print()
    if ok and n_delta <= n_full:
        print(
            "PASS: incremental reconstruction is byte-exact, delta is lossless, "
            f"and delta saved {n_full - n_delta:,} bytes "
            f"({100 * (1 - n_delta / n_full):.1f}%)."
        )
        sys.exit(0)
    if not ok:
        print("FAIL: incremental, full, or delta reconstructions differ.")
    else:
        print("FAIL: delta bitstream is LARGER than full (never-worse violated).")
    sys.exit(1)


if __name__ == "__main__":
    main()
