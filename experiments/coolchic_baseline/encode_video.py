#!/usr/bin/env python
"""Robust replacement for Cool-Chic's samples/encode.py.

Same per-frame configs as samples/encode.py (Cool-Chic 5.0.1), but:
- uses sys.executable instead of bare "python3"
- runs _getcodingstruct.py with stderr separated and return-code checked
- locates the TSV header line instead of assuming it is the first line

Run inside the coolchic env:
  python encode_video.py --coolchic ~/Cool-Chic -i in.yuv -o bitstream.cool \
      --workdir work --n_frames 32 --lmbda 0.02
"""
import argparse
import subprocess
import sys
from pathlib import Path


def frame_config(cfg: Path, ftype: str, depth: int, lmbda: float,
                 starved: bool = False) -> list[str]:
    if ftype == "I":
        return [
            f"--dec_cfg_residue={cfg}/dec/intra/hop.cfg",
            "--start_lr=1e-2",
            "--n_itr=10000",
            f"--lmbda={lmbda}",
        ]
    if ftype == "P":
        if starved:
            # Pose-carrier frame (even display index under --tune=comma):
            # judged only by PoseNet, which reads ego-motion — so keep a good
            # motion model but starve the residual (lightest residual config +
            # high rate pressure). The warp of the previous frame carries the
            # motion signal almost for free.
            return [
                f"--dec_cfg_residue={cfg}/dec/residue/lop.cfg",
                f"--dec_cfg_motion={cfg}/dec/motion/mop.cfg",
                "--start_lr=5e-3",
                "--n_itr_pretrain_motion=3000",
                "--n_itr=10000",
                f"--lmbda={lmbda}",
            ]
        return [
            f"--dec_cfg_residue={cfg}/dec/residue/mop.cfg",
            f"--dec_cfg_motion={cfg}/dec/motion/mop.cfg",
            "--start_lr=5e-3",
            "--n_itr_pretrain_motion=3000",
            "--n_itr=10000",
            f"--lmbda={lmbda}",
        ]
    # B-frame: lighter configs deeper in the GOP, lambda scaled by depth
    n_itr = max(10000 - 2000 * depth, 1000)
    n_itr_motion = max(5000 - 1000 * depth, 1000)
    sub = "mop" if depth == 1 else "lop"
    return [
        f"--n_itr_pretrain_motion={n_itr_motion}",
        f"--n_itr={n_itr}",
        f"--lmbda={1.5 ** depth * lmbda}",
        f"--dec_cfg_residue={cfg}/dec/residue/{sub}.cfg",
        f"--dec_cfg_motion={cfg}/dec/motion/{sub}.cfg",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coolchic", required=True, type=Path, help="Cool-Chic repo root")
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--n_frames", type=int, required=True)
    ap.add_argument("--intra_pos", default="0")
    ap.add_argument("--p_pos", default="",
                    help="empty = auto: P every --gop frames and on the last frame "
                         "(Cool-Chic requires the last frame to be I or P)")
    ap.add_argument("--gop", type=int, default=16)
    ap.add_argument("--lmbda", type=float, required=True)
    ap.add_argument("--even_lmbda_mult", type=float, default=8.0,
                    help="rate-pressure multiplier for even (PoseNet-only) "
                         "P-frames under --tune=comma; 1.0 disables starvation")
    ap.add_argument("--extra_args", default="")
    args = ap.parse_args()

    cc = args.coolchic.expanduser().resolve()
    cfg = cc / "cfg"
    py = sys.executable

    if not args.p_pos and args.n_frames > 1:
        p_positions = list(range(args.gop, args.n_frames - 1, args.gop))
        p_positions.append(args.n_frames - 1)
        args.p_pos = ",".join(str(p) for p in p_positions)
        print(f"auto p_pos: {args.p_pos}", flush=True)

    struct_args = [
        f"--intra_pos={args.intra_pos}",
        f"--p_pos={args.p_pos}",
        f"--n_frames={args.n_frames}",
    ]
    r = subprocess.run(
        [py, str(cc / "_getcodingstruct.py"), *struct_args, "--raw_coding_struct"],
        capture_output=True, text=True, cwd=str(cc),
    )
    if r.returncode != 0:
        sys.exit(f"_getcodingstruct.py failed (rc={r.returncode}):\n"
                 f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    try:
        h = next(i for i, ln in enumerate(lines) if ln.startswith("coding\t"))
    except StopIteration:
        sys.exit(f"no coding-structure header in output:\n"
                 f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    header = lines[h].split("\t")
    frames = [dict(zip(header, ln.split("\t"))) for ln in lines[h + 1:]]
    frames = frames[: args.n_frames]
    if len(frames) != args.n_frames:
        sys.exit(f"coding struct has {len(frames)} frames, expected {args.n_frames}")

    print(f"coding structure: "
          + " ".join(f"{f['type']}{f['display']}" for f in frames), flush=True)

    # Resume support. A frame is complete iff BOTH exist in the workdir:
    #   XXXX-decoded-<seq>.yuv     (reference for later frames; saved first)
    #   XXXX-results_decoder.tsv   (written AFTER the bitstream append, so its
    #                               presence proves the frame's chunk landed)
    seq_name = Path(args.input).name.rsplit(".", 1)[0]
    w_str, h_str = seq_name.split("_")[1].split("x")
    frame_yuv_bytes = int(w_str) * int(h_str) * 3 // 2
    workdir = Path(args.workdir)

    comma_mode = "comma" in args.extra_args

    for coding_idx, fr in enumerate(frames):
        ftype, depth = fr["type"], int(fr["depth"])
        display = int(fr["display"])
        prefix = f"{display:04d}-"
        dec = workdir / f"{prefix}decoded-{seq_name}.yuv"
        tsv = workdir / f"{prefix}results_decoder.tsv"
        if dec.exists() and dec.stat().st_size == frame_yuv_bytes and tsv.exists():
            print(f"[frame {coding_idx + 1}/{len(frames)}] already encoded "
                  f"(found {tsv.name}), skipping", flush=True)
            continue
        # Idea 1: even display index (PoseNet-only) P-frames become starved
        # pose carriers — lightest residual + even_lmbda_mult x rate pressure.
        starved = (comma_mode and ftype == "P" and display % 2 == 0
                   and args.even_lmbda_mult != 1.0)
        frame_lmbda = args.lmbda * (args.even_lmbda_mult if starved else 1.0)
        cmd = [
            py, str(cc / "cc_encode.py"),
            f"--input={args.input}",
            f"--output={args.output}",
            f"--workdir={args.workdir}",
            *struct_args,
            f"--coding_idx={coding_idx}",
            *frame_config(cfg, ftype, depth, frame_lmbda, starved=starved),
        ]
        if args.extra_args:
            cmd += args.extra_args.split()
        role = "pose-carrier(starved)" if starved else (
            "judged(odd)" if comma_mode and display % 2 == 1 else "std")
        print(f"\n[frame {coding_idx + 1}/{len(frames)}] coding_idx={coding_idx} "
              f"type={ftype} depth={depth} display={display} role={role} "
              f"lmbda={frame_lmbda:g}", flush=True)
        subprocess.run(cmd, cwd=str(cc), check=True)

    print("\nencode complete:", args.output, flush=True)


if __name__ == "__main__":
    main()
