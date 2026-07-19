#!/usr/bin/env python
"""Score a Cool-Chic reconstruction against the challenge metric.

Run in the *comma-compress* env (needs torch, av, timm, smp...), from anywhere:

  python experiments/coolchic_baseline/score_coolchic.py \
      --run experiments/coolchic_baseline/runs/lmbda_0.02_n200 [--device cuda]

Reads <run>/recon_1164x874_20p_yuv420_8b.yuv and <run>/bitstream.cool,
compares the first N reconstructed frames against the ground-truth decode of
videos/0.mkv (the exact harness path: PyAV + frame_utils.yuv420_to_rgb), and
prints seg/pose/rate/score. Rate is reported two ways: raw bitstream bytes
over the full 37,545,489-byte denominator, and extrapolated to 1200 frames.
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


def read_recon_yuv(path: Path, n_frames: int):
    """Yield uint8 HWC RGB tensors (cropped to camera size) from the padded
    raw yuv420p reconstruction."""
    frame_bytes = PAD_W * PAD_H * 3 // 2
    data = np.memmap(path, dtype=np.uint8, mode="r")
    n_avail = data.size // frame_bytes
    if n_avail < n_frames:
        raise SystemExit(f"recon has {n_avail} frames, expected {n_frames}")
    for i in range(n_frames):
        off = i * frame_bytes
        y = np.array(data[off : off + PAD_W * PAD_H]).reshape(PAD_H, PAD_W)
        off += PAD_W * PAD_H
        u = np.array(data[off : off + PAD_W * PAD_H // 4]).reshape(PAD_H // 2, PAD_W // 2)
        off += PAD_W * PAD_H // 4
        v = np.array(data[off : off + PAD_W * PAD_H // 4]).reshape(PAD_H // 2, PAD_W // 2)
        rgb = yuv_planes_to_rgb(y, u, v)  # (PAD_H, PAD_W, 3)
        yield rgb[:H, :W, :].contiguous()


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
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    recon_path = args.run / "recon_1168x880_20p_yuv420_8b.yuv"
    bitstream_path = args.run / "bitstream.cool"
    n_frames = int(np.memmap(recon_path, dtype=np.uint8, mode="r").size // (PAD_W * PAD_H * 3 // 2))
    n_pairs = n_frames // 2
    print(f"run: {args.run}  ({n_frames} frames = {n_pairs} pairs, device {device})")

    net = DistortionNet().eval().to(device)
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

    gt = read_gt_frames(n_frames)
    recon = list(read_recon_yuv(recon_path, n_frames))

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

    bytes_actual = bitstream_path.stat().st_size
    bytes_extrap = bytes_actual * TOTAL_FRAMES / n_frames
    rate_actual = bytes_actual / UNCOMPRESSED_BYTES
    rate_extrap = bytes_extrap / UNCOMPRESSED_BYTES

    score_extrap = 100 * seg + math.sqrt(10 * pose) + 25 * rate_extrap
    print(f"  seg  = {seg:.8f}   (100*seg = {100*seg:.4f})")
    print(f"  pose = {pose:.8f}   (sqrt(10*pose) = {math.sqrt(10*pose):.4f})")
    print(f"  bitstream = {bytes_actual:,} B for {n_frames} frames "
          f"-> {bytes_extrap:,.0f} B extrapolated to {TOTAL_FRAMES}")
    print(f"  rate = {rate_actual:.6f} raw, {rate_extrap:.6f} extrapolated "
          f"(25*rate_extrap = {25*rate_extrap:.4f})")
    print(f"  SCORE (extrapolated rate) = {score_extrap:.4f}")
    print(f"  [refs: ffmpeg baseline 4.39 | hnerv_muon ~0.20 | SOTA 0.1885]")

    with open(HERE / "results.csv", "a") as f:
        f.write(f"{args.run.name},{n_frames},{bytes_actual},{seg:.8f},{pose:.8f},"
                f"{rate_extrap:.8f},{score_extrap:.6f}\n")
    print(f"appended to {HERE / 'results.csv'}")


if __name__ == "__main__":
    main()
