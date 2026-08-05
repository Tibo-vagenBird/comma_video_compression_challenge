#!/usr/bin/env python
"""Score a Cool-Chic reconstruction against the challenge metric.

Run in the *comma-compress* env (needs torch, av, timm, smp...), from anywhere:

  python experiments/coolchic_baseline/score_coolchic.py \
      --run experiments/coolchic_baseline/runs/lmbda_0.02_n200 [--device cuda]

Reads <run>/recon_1164x874_20p_yuv420_8b.yuv and <run>/bitstream.cool,
compares the first N reconstructed frames against the ground-truth decode of
videos/0.mkv (the exact harness path: PyAV + frame_utils.yuv420_to_rgb), and
prints seg/pose/rate/score. The score uses a temporary Deflate-9 ZIP containing
the bitstream, matching the challenge's outer-archive rate accounting. Raw
bitstream bytes remain visible as a diagnostic.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

import av  # noqa: E402
from frame_utils import yuv420_to_rgb, camera_size  # noqa: E402
from modules import DistortionNet, segnet_sd_path, posenet_sd_path  # noqa: E402
from experiments.coolchic_baseline.archive_rate import (  # noqa: E402
    append_results,
    calculate_archive_score,
)

W, H = camera_size  # (1164, 874)
PAD_W, PAD_H = 1168, 880  # encode-side padding for RAFT (needs /8); cropped here
TOTAL_FRAMES = 1200
UNCOMPRESSED_BYTES = 37_545_489  # videos/0.mkv, the official rate denominator


def yuv_planes_to_rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> torch.Tensor:
    """Exact reimplementation of frame_utils.yuv420_to_rgb for raw planes:
    BT.601 limited range, bilinear chroma upsampling, round, uint8 HWC."""
    h, w = y.shape
    y_t = torch.from_numpy(y.astype(np.float32))
    u_t = torch.from_numpy(u.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    v_t = torch.from_numpy(v.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    u_up = F.interpolate(u_t, size=(h, w), mode="bilinear", align_corners=False).squeeze()
    v_up = F.interpolate(v_t, size=(h, w), mode="bilinear", align_corners=False).squeeze()

    yf = (y_t - 16.0) * (255.0 / 219.0)
    uf = (u_up - 128.0) * (255.0 / 224.0)
    vf = (v_up - 128.0) * (255.0 / 224.0)

    r = (yf + 1.402 * vf).clamp(0, 255)
    g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255)
    b = (yf + 1.772 * uf).clamp(0, 255)
    return torch.stack([r, g, b], dim=-1).round().to(torch.uint8)


def read_recon_yuv(path: Path, n_frames: int, rw: int, rh: int):
    """Yield uint8 HWC RGB tensors at camera size from a raw yuv420p recon.

    Full-res recon (1168x880): convert then crop the padding off.
    Eval-res recon (512x384): convert then bicubic-upsample to camera size,
    mirroring the intended inflate chain of a 384x512-coded submission.
    """
    frame_bytes = rw * rh * 3 // 2
    data = np.memmap(path, dtype=np.uint8, mode="r")
    n_avail = data.size // frame_bytes
    if n_avail < n_frames:
        raise SystemExit(f"recon has {n_avail} frames, expected {n_frames}")
    for i in range(n_frames):
        off = i * frame_bytes
        y = np.array(data[off : off + rw * rh]).reshape(rh, rw)
        off += rw * rh
        u = np.array(data[off : off + rw * rh // 4]).reshape(rh // 2, rw // 2)
        off += rw * rh // 4
        v = np.array(data[off : off + rw * rh // 4]).reshape(rh // 2, rw // 2)
        rgb = yuv_planes_to_rgb(y, u, v)  # (rh, rw, 3) uint8
        if (rh, rw) == (PAD_H, PAD_W):
            yield rgb[:H, :W, :].contiguous()
        elif (rh, rw) == (H, W):
            yield rgb
        else:
            up = F.interpolate(
                rgb.float().permute(2, 0, 1).unsqueeze(0),
                size=(H, W), mode="bicubic", align_corners=False,
            ).clamp(0, 255).round().to(torch.uint8)
            yield up.squeeze(0).permute(1, 2, 0).contiguous()


def read_gt_frames(n_frames: int):
    """Ground truth via the exact harness decode path."""
    container = av.open(str(ROOT / "videos" / "0.mkv"))
    out = []
    for frame in container.decode(container.streams.video[0]):
        out.append(yuv420_to_rgb(frame))
        if len(out) == n_frames:
            break
    container.close()
    if len(out) < n_frames:
        raise SystemExit(f"GT video has only {len(out)} frames, expected {n_frames}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="run dir from run_one.sh")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch-pairs", type=int, default=8)
    ap.add_argument("--diagnose", action="store_true",
                    help="pose root-cause probes (reuses recon, no retrain)")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    recon_path = None
    for geom in ((PAD_W, PAD_H), (512, 384)):
        cand = args.run / f"recon_{geom[0]}x{geom[1]}_20p_yuv420_8b.yuv"
        if cand.exists():
            recon_path, (rw, rh) = cand, geom
            break
    if recon_path is None:
        raise SystemExit(f"no recon_*.yuv found in {args.run}")
    bitstream_path = args.run / "bitstream.cool"
    n_frames = int(np.memmap(recon_path, dtype=np.uint8, mode="r").size // (rw * rh * 3 // 2))
    n_pairs = n_frames // 2
    print(f"run: {args.run}  ({n_frames} frames = {n_pairs} pairs, device {device})")

    net = DistortionNet().eval().to(device)
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

    gt = read_gt_frames(n_frames)
    recon = list(read_recon_yuv(recon_path, n_frames, rw, rh))

    seg_sum, pose_sum = 0.0, 0.0
    with torch.inference_mode():
        for start in range(0, n_pairs, args.batch_pairs):
            idx = range(start, min(start + args.batch_pairs, n_pairs))
            gt_b = torch.stack([torch.stack([gt[2 * p], gt[2 * p + 1]]) for p in idx]).to(device)
            rc_b = torch.stack([torch.stack([recon[2 * p], recon[2 * p + 1]]) for p in idx]).to(device)
            pose_d, seg_d = net.compute_distortion(gt_b, rc_b)
            seg_sum += seg_d.sum().item()
            pose_sum += pose_d.sum().item()

    seg = seg_sum / n_pairs
    pose = pose_sum / n_pairs

    result = calculate_archive_score(
        bitstream_path=bitstream_path,
        n_frames=n_frames,
        total_frames=TOTAL_FRAMES,
        uncompressed_bytes=UNCOMPRESSED_BYTES,
        seg=seg,
        pose=pose,
    )
    print(f"  seg  = {seg:.8f}   (100*seg = {100*seg:.4f})")
    print(f"  pose = {pose:.8f}   (sqrt(10*pose) = {math.sqrt(10*pose):.4f})")
    print(f"  bitstream = {result.raw_bytes:,} B raw")
    print(f"  temporary ZIP = {result.archive_bytes:,} B for {n_frames} frames "
          f"-> {result.archive_bytes_extrap:,.0f} B extrapolated to {TOTAL_FRAMES}")
    print(f"  ZIP rate = {result.rate_actual:.6f} partial, "
          f"{result.rate_extrap:.6f} extrapolated "
          f"(25*rate_extrap = {25*result.rate_extrap:.4f})")
    print(f"  SCORE (extrapolated rate) = {result.score:.4f}")
    print(f"  [refs: ffmpeg baseline 4.39 | hnerv_muon ~0.20 | SOTA 0.1885]")

    results_path = HERE / "results_v2.csv"
    append_results(
        results_path,
        args.run.name,
        n_frames,
        result.raw_bytes,
        result.archive_bytes,
        seg,
        pose,
        result.rate_extrap,
        result.score,
    )
    print(f"appended to {results_path}")

    if args.diagnose:
        run_diagnostics(net, gt, recon, n_pairs, device)
        run_seg_diagnostics(net, gt, recon, n_pairs, device)


def _seg_labels(net, pair, device):
    """SegNet argmax label map [384,512] of a pair's last (judged) frame."""
    b = torch.stack([torch.stack(pair)]).to(device)  # (1,2,H,W,3)
    _, seg_logits = net(b)          # DistortionNet.forward -> (pose, seg logits)
    return seg_logits.argmax(dim=1).squeeze(0)       # (384,512) long


def run_seg_diagnostics(net, gt, recon, n_pairs, device):
    print("\n=== SEG DIAGNOSTICS ===")
    idxs = list(range(n_pairs))
    H, W = 384, 512

    per_pair = []
    n_class = 5
    # pixels that FLIP, bucketed by their GROUND-TRUTH class and by whether the
    # GT pixel sits on a class boundary; plus a confusion count gt_c -> rc_c.
    flip_by_gtclass = np.zeros(n_class)
    total_by_gtclass = np.zeros(n_class)
    confusion = np.zeros((n_class, n_class))
    band_flip = np.zeros(4)          # 4 vertical bands, top(sky)->bottom(road)
    band_total = np.zeros(4)
    boundary_flip = interior_flip = boundary_total = interior_total = 0

    with torch.inference_mode():
        for p in idxs:
            gtl = _seg_labels(net, (gt[2 * p], gt[2 * p + 1]), device)
            rcl = _seg_labels(net, (recon[2 * p], recon[2 * p + 1]), device)
            flip = (gtl != rcl)
            per_pair.append(flip.float().mean().item())

            g = gtl.cpu().numpy(); r = rcl.cpu().numpy(); f = flip.cpu().numpy()
            for c in range(n_class):
                m = (g == c)
                total_by_gtclass[c] += m.sum()
                flip_by_gtclass[c] += (f & m).sum()
            gv, rv, fv = g[f], r[f], None
            for gc, rc in zip(gv, rv):
                confusion[gc, rc] += 1

            bands = np.array_split(np.arange(H), 4)
            for bi, rows in enumerate(bands):
                band_total[bi] += f[rows].size
                band_flip[bi] += f[rows].sum()

            # boundary = GT pixel whose 4-neighborhood has any other GT class
            b = np.zeros_like(g, dtype=bool)
            b[:-1] |= g[:-1] != g[1:]; b[1:] |= g[1:] != g[:-1]
            b[:, :-1] |= g[:, :-1] != g[:, 1:]; b[:, 1:] |= g[:, 1:] != g[:, :-1]
            boundary_total += b.sum(); interior_total += (~b).sum()
            boundary_flip += (f & b).sum(); interior_flip += (f & ~b).sum()

    arr = np.array(per_pair)
    print(f"(A) per-pair seg flip-rate: min {arr.min():.5f}  med {np.median(arr):.5f}  "
          f"max {arr.max():.5f}  mean {arr.mean():.5f}")
    worst = arr.argsort()[::-1][:5]
    print(f"    worst pairs: {[(int(i), round(float(arr[i]), 5)) for i in worst]}")

    print("(B) flip rate by GROUND-TRUTH class (which classes are unstable):")
    for c in range(n_class):
        if total_by_gtclass[c] > 0:
            print(f"    class {c}: {flip_by_gtclass[c]/total_by_gtclass[c]:.5f} flip"
                  f"  ({total_by_gtclass[c]/total_by_gtclass.sum()*100:4.1f}% of pixels,"
                  f"  {flip_by_gtclass[c]/flip_by_gtclass.sum()*100:4.1f}% of all flips)")

    print("(C) confusion of FLIPPED pixels (gt_class -> recon_class), row-normalized:")
    for c in range(n_class):
        row = confusion[c]
        if row.sum() > 0:
            frac = row / row.sum()
            print(f"    gt {c} -> " + "  ".join(f"{j}:{frac[j]:.2f}" for j in range(n_class)))

    print("(D) spatial concentration (4 vertical bands, top=sky ... bottom=road):")
    for bi in range(4):
        if band_total[bi] > 0:
            print(f"    band {bi} ({'top' if bi==0 else 'bottom' if bi==3 else 'mid'}): "
                  f"{band_flip[bi]/band_total[bi]:.5f} flip  "
                  f"({band_flip[bi]/band_flip.sum()*100:4.1f}% of all flips)")

    bf = boundary_flip / max(boundary_total, 1)
    itf = interior_flip / max(interior_total, 1)
    print(f"(E) boundary vs interior: boundary {bf:.5f} flip "
          f"({boundary_flip/(boundary_flip+interior_flip)*100:.1f}% of flips, "
          f"{boundary_total/(boundary_total+interior_total)*100:.1f}% of pixels) | "
          f"interior {itf:.5f} flip")
    print(f"    -> flips are {'BOUNDARY-dominated (need sharper edges)' if bf > 3*itf else 'spread through interiors (capacity problem)'}")
    print("=== END SEG DIAGNOSTICS ===")


def _pose_pairs(net, frame0_list, frame1_list, idxs, device):
    """Per-pair PoseNet distortion for the given (frame0, frame1) lists."""
    out = []
    with torch.inference_mode():
        for p in idxs:
            gt0, gt1 = frame0_list[p]
            rc0, rc1 = frame1_list[p]
            gt_b = torch.stack([torch.stack([gt0, gt1])]).to(device)
            rc_b = torch.stack([torch.stack([rc0, rc1])]).to(device)
            pose_d, _ = net.compute_distortion(gt_b, rc_b)
            out.append(pose_d.item())
    return out


def run_diagnostics(net, gt, recon, n_pairs, device):
    print("\n=== POSE DIAGNOSTICS ===")
    idxs = list(range(n_pairs))

    # (A) Per-pair pose: uniform (systematic) vs spiky (a few bad pairs)?
    gt_pairs = [(gt[2 * p], gt[2 * p + 1]) for p in idxs]
    rc_pairs = [(recon[2 * p], recon[2 * p + 1]) for p in idxs]
    per = _pose_pairs(net, gt_pairs, rc_pairs, idxs, device)
    arr = np.array(per)
    print(f"(A) per-pair pose: min {arr.min():.3f}  med {np.median(arr):.3f}  "
          f"max {arr.max():.3f}  mean {arr.mean():.3f}")
    worst = arr.argsort()[::-1][:5]
    print(f"    worst pairs: {[(int(i), round(float(arr[i]), 2)) for i in worst]}")

    # (B) Alignment: recompute pose with recon shifted by +/-1 frame. If a
    #     shift collapses pose, our pairing/frame order is off by one.
    def shifted(shift):
        rp = []
        for p in idxs:
            a, b = 2 * p + shift, 2 * p + 1 + shift
            if 0 <= a < 2 * n_pairs and 0 <= b < 2 * n_pairs:
                rp.append((recon[a], recon[b]))
            else:
                rp.append((recon[2 * p], recon[2 * p + 1]))
        return rp
    for s in (-1, 1):
        ps = _pose_pairs(net, gt_pairs, shifted(s), idxs, device)
        print(f"(B) recon shifted {s:+d}: mean pose {np.mean(ps):.3f} "
              f"(vs aligned {arr.mean():.3f})")

    # (C) Scale reference: pose of degenerate recons vs GT, to know what
    #     "large" means for this metric on this clip.
    #   c1: recon = GT (should be ~0; nonzero = numeric floor)
    id_pose = _pose_pairs(net, gt_pairs, gt_pairs, idxs, device)
    #   c2: recon = GT frame0 duplicated (kills all motion) -> pose of "no motion"
    dup_pairs = [(gt[2 * p], gt[2 * p]) for p in idxs]
    dup_pose = _pose_pairs(net, gt_pairs, dup_pairs, idxs, device)
    print(f"(C) pose[recon=GT] mean {np.mean(id_pose):.4f} (numeric floor)")
    print(f"    pose[recon=GT frame0 duplicated] mean {np.mean(dup_pose):.3f} "
          f"(what 'zero motion' costs)")
    print(f"    -> our pose {arr.mean():.3f} vs no-motion {np.mean(dup_pose):.3f}: "
          f"{'WORSE than no-motion (recon injects false motion)' if arr.mean() > np.mean(dup_pose) else 'better than no-motion'}")
    print("=== END DIAGNOSTICS ===")


if __name__ == "__main__":
    main()
