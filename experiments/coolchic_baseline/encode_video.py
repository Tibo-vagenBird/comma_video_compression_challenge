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
import os
import subprocess
import sys
from pathlib import Path


def _fmt_onoff(on: bool) -> str:
    return "ON " if on else "off"


def _print_lever_banner(args, cc: Path, comma_mode: bool) -> None:
    """Print one consolidated block of every active lever/idea at the top of the
    log, so encode.log opens with the exact config instead of scattered lines.

    Judge-loss parameters live in comma.py (env-driven with defaults). We import
    its already-resolved module constants so the banner can never drift from the
    real defaults; if that import fails we fall back to reading the env with the
    same documented defaults. Importing comma.py does NOT load the judges (that
    is lazy), so this is cheap and side-effect free.
    """
    cm = None
    if comma_mode:
        try:
            if str(cc) not in sys.path:
                sys.path.insert(0, str(cc))
            from coolchic.training.metrics import comma as _cm
            cm = _cm
        except Exception as e:  # a display helper must never break the encode
            print(f"[banner] comma metric import failed ({e}); using env defaults",
                  flush=True)

    # env name -> (comma.py attribute, default string matching comma.py)
    _CD = {
        "COMMA_SEG_W": ("_SEG_W", "100.0"),
        "COMMA_POSE_W": ("_POSE_W", "1.0"),
        "COMMA_EVEN_MSE_W": ("_EVEN_MSE_W", "1.0"),
        "COMMA_ODD_MSE_W": ("_ODD_MSE_W", "0.0"),
        "COMMA_SEG_TAU": ("_SEG_TAU", "0.3"),
        "COMMA_L7_MULT": ("_L7_MULT", "4.0"),
        "COMMA_L7_THRESH": ("_L7_THRESH", "1.0"),
        "COMMA_SEG_CLASS_POW": ("_SEG_CLASS_POW", "0.0"),
        "COMMA_WARMUP_MSE_ITERS": ("_WARMUP_MSE_ITERS", "0"),
        "COMMA_JUDGE_AMP": ("_JUDGE_AMP", "0"),
    }

    def cv(env_name):
        attr, default = _CD[env_name]
        if cm is not None:
            return getattr(cm, attr)
        return os.environ.get(env_name, default)

    def as_float(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0

    def truthy(x):
        return (x is True) or (str(x).strip() in ("1", "True", "true"))

    bar = "=" * 80
    out = [bar, "  COOL-CHIC x COMMA -- ACTIVE LEVERS & PARAMETERS", bar,
           "  Core RD",
           f"    lmbda ............................. {args.lmbda:g}",
           f"    n_frames .......................... {args.n_frames}"
           f"   (intra_pos={args.intra_pos}, p_pos={args.p_pos or 'auto'})",
           f"    tune .............................. {'comma' if comma_mode else 'mse'}",
           f"    seed .............................. {os.environ.get('COMMA_SEED', '1234')}"
           f"   (base; per-frame = base + coding_idx)"]

    if comma_mode:
        out += ["  Judge losses",
                f"    seg_w      COMMA_SEG_W ............ {cv('COMMA_SEG_W')}",
                f"    pose_w     COMMA_POSE_W .......... {cv('COMMA_POSE_W')}",
                f"    even_mse_w COMMA_EVEN_MSE_W ...... {cv('COMMA_EVEN_MSE_W')}",
                f"    odd_mse_w  COMMA_ODD_MSE_W ....... {cv('COMMA_ODD_MSE_W')}",
                f"    seg_tau    COMMA_SEG_TAU ......... {cv('COMMA_SEG_TAU')}"]
        starve_on = args.even_lmbda_mult != 1.0
        out.append(f"  Idea 1 -- even-frame starvation ..... {_fmt_onoff(starve_on)}"
                   f"  (even_lmbda_mult={args.even_lmbda_mult:g})")
        _l7_cap = os.environ.get("COMMA_L7_CAP", "1e9")
        _l7_cap_on = as_float(_l7_cap) < 1e8
        out.append(f"  Idea 4 -- L7 boundary weighting ..... "
                   f"{_fmt_onoff(as_float(cv('COMMA_L7_MULT')) > 0)}"
                   f"  (l7_mult={cv('COMMA_L7_MULT')} l7_thresh={cv('COMMA_L7_THRESH')}"
                   f" swing_cap={_l7_cap if _l7_cap_on else 'off'})")
        out.append(f"  Seg class reweighting ............... "
                   f"{_fmt_onoff(as_float(cv('COMMA_SEG_CLASS_POW')) > 0)}"
                   f"  (seg_class_pow={cv('COMMA_SEG_CLASS_POW')})")

    ws = bool(args.warmstart)
    _wu_noise = (f"soft-gauss({os.environ.get('WARMSTART_WARMUP_NOISE', '0.25')})"
                 if os.environ.get("WARMSTART_WARMUP_SOFT", "0") == "1" else "kuma")
    ws_detail = (f"  (n_itr={args.warmstart_n_itr} motion={args.warmstart_n_itr_motion} "
                 f"warmup_cand={os.environ.get('WARMSTART_WARMUP_CAND', '1')} "
                 f"warmup_itr={os.environ.get('WARMSTART_WARMUP_ITR', '200')} "
                 f"warmup_noise={_wu_noise})" if ws else "")
    out.append(f"  Lever 2 -- temporal warm-start ...... {_fmt_onoff(ws)}{ws_detail}")
    _skip_dec = os.environ.get("SKIP_PERFRAME_DECODE", "1") == "1"
    out.append(f"  Per-frame re-decode ................. "
               f"{'SKIPPED (tsv from stored recon)' if _skip_dec else 'ON  (O(N^2) -- slow at scale)'}")
    _delta_on = os.environ.get("COMMA_DELTA", "0") == "1"
    out.append(f"  Delta weight-coding COMMA_DELTA ..... "
               f"{_fmt_onoff(_delta_on)}"
               f"  (Phase 1.5 re-expression; same-role ref, lossless)")

    if comma_mode:
        amp_dtype = os.environ.get("COMMA_JUDGE_AMP_DTYPE", "bf16").lower()
        amp_dtype = "fp16" if amp_dtype in ("fp16", "float16") else "bf16"
        out.append(f"  Lever A -- {amp_dtype} judges (AMP) ........ "
                   f"{_fmt_onoff(truthy(cv('COMMA_JUDGE_AMP')))}")
        out.append(f"  Lever 3 -- MSE-warmup ............... "
                   f"{_fmt_onoff(as_float(cv('COMMA_WARMUP_MSE_ITERS')) > 0)}"
                   f"  (iters={cv('COMMA_WARMUP_MSE_ITERS')}; harmful, keep off)")

    out.append(bar)
    print("\n".join(out), flush=True)


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
    ap.add_argument("--even_lmbda_mult", type=float, default=4.0,
                    help="rate-pressure multiplier for even (PoseNet-only) "
                         "P-frames under --tune=comma; 1.0 disables starvation. "
                         "Default 4 (eased from 8: 8x over-starved high-motion "
                         "pairs and hurt pose)")
    ap.add_argument("--warmstart", action="store_true",
                    help="temporal warm-start: init each frame's encoder from the "
                         "previous coded frame and cut its iteration count")
    ap.add_argument("--warmstart_n_itr", type=int, default=3000,
                    help="main iterations for warm-started frames (vs 10000 cold)")
    ap.add_argument("--warmstart_n_itr_motion", type=int, default=1000,
                    help="motion-pretrain iterations for warm-started frames")
    ap.add_argument("--extra_args", default="")
    args = ap.parse_args()

    cc = args.coolchic.expanduser().resolve()
    cfg = cc / "cfg"
    py = sys.executable

    comma_mode = "comma" in args.extra_args
    _print_lever_banner(args, cc, comma_mode)

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

        # Lever 2: temporal warm-start. Init from the previous coded frame's
        # encoder and cut the iteration count (appended last so the reduced
        # --n_itr overrides frame_config's value under configargparse).
        warm = False
        if args.warmstart and coding_idx > 0:
            # Same-ROLE warm-start: init from the most recent previous frame
            # with the SAME type AND same display parity (odd<-odd, even<-even).
            # Under --tune=comma with starvation, even frames are degenerate
            # pose-carriers; warm-starting a judged (odd) frame from a carrier
            # poisons it (SegNet totally disagrees -> saturated gradient ->
            # stuck). Same-role also guarantees a full architecture match
            # (same dec cfg), not a partial load.
            src = None
            for j in range(coding_idx - 1, -1, -1):
                pj = frames[j]
                if pj["type"] == fr["type"] and int(pj["display"]) % 2 == display % 2:
                    src = pj
                    break
            if src is not None:
                prev_pt = workdir / f"{int(src['display']):04d}-frame_encoder.pt"
                if prev_pt.is_file():
                    cmd += [f"--warmstart_from={prev_pt}",
                            f"--n_itr={args.warmstart_n_itr}",
                            f"--n_itr_pretrain_motion={args.warmstart_n_itr_motion}"]
                    warm = True

        role = "pose-carrier(starved)" if starved else (
            "judged(odd)" if comma_mode and display % 2 == 1 else "std")
        print(f"\n[frame {coding_idx + 1}/{len(frames)}] coding_idx={coding_idx} "
              f"type={ftype} depth={depth} display={display} role={role} "
              f"lmbda={frame_lmbda:g} warmstart={warm}", flush=True)
        subprocess.run(cmd, cwd=str(cc), check=True)

    print("\nencode complete:", args.output, flush=True)


if __name__ == "__main__":
    main()
