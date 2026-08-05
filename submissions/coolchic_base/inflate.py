#!/usr/bin/env python
"""Inflate the coolchic_base submission to the challenge's raw-frame format.

Decode chain (reference, matches the encode that produced the 1.33 result):
  Cool-Chic bitstream  --cc_decode.py-->  YUV420 @ 512x384
    -> RGB  (BT.601 limited range, bilinear chroma upsample; identical to the
             harness frame_utils.yuv420_to_rgb)
    -> bicubic upsample 512x384 -> 874x1164
    -> uint8 HWC, streamed to <base>.raw

Requires the Cool-Chic 5.0.1 stack (torch>=2.11, constriction, the coolchic
package). Point COOLCHIC_REPO at the Cool-Chic checkout (default: ~/Cool-Chic).

NOTE: the shipped archive.zip is a 32-frame proof-of-concept (see README.md).
A full submission requires encoding all 1200 frames of the video.

Usage (challenge harness):  inflate.py <archive_dir> <inflated_dir> <names_file>
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

W, H = 1164, 874           # camera (output) size, (W, H)
EVAL_W, EVAL_H = 512, 384  # coded resolution


def yuv_planes_to_rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> torch.Tensor:
    """YUV420 planes -> (h, w, 3) uint8 RGB. BT.601 limited range, bilinear
    chroma upsample — identical math to frame_utils.yuv420_to_rgb."""
    h, w = y.shape
    y_t = torch.from_numpy(y.astype(np.float32))
    u_t = torch.from_numpy(u.astype(np.float32))[None, None]
    v_t = torch.from_numpy(v.astype(np.float32))[None, None]
    u_up = F.interpolate(u_t, size=(h, w), mode="bilinear", align_corners=False).squeeze()
    v_up = F.interpolate(v_t, size=(h, w), mode="bilinear", align_corners=False).squeeze()
    yf = (y_t - 16.0) * (255.0 / 219.0)
    uf = (u_up - 128.0) * (255.0 / 224.0)
    vf = (v_up - 128.0) * (255.0 / 224.0)
    r = (yf + 1.402 * vf).clamp(0, 255)
    g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255)
    b = (yf + 1.772 * uf).clamp(0, 255)
    return torch.stack([r, g, b], dim=-1).round().to(torch.uint8)


def upsample_to_camera(rgb: torch.Tensor) -> torch.Tensor:
    """(EVAL_H, EVAL_W, 3) uint8 -> (H, W, 3) uint8 via bicubic (decode chain)."""
    up = F.interpolate(
        rgb.float().permute(2, 0, 1).unsqueeze(0),
        size=(H, W), mode="bicubic", align_corners=False,
    ).clamp(0, 255).round().to(torch.uint8)
    return up.squeeze(0).permute(1, 2, 0).contiguous()


def decode_bitstream(bitstream: Path, out_yuv: Path, coolchic: Path) -> None:
    subprocess.run(
        [sys.executable, str(coolchic / "cc_decode.py"),
         "-i", str(bitstream), "-o", str(out_yuv)],
        check=True,
    )


def yuv_to_raw(recon_yuv: Path, dst_raw: Path) -> int:
    frame_bytes = EVAL_W * EVAL_H * 3 // 2
    data = np.memmap(recon_yuv, dtype=np.uint8, mode="r")
    n = data.size // frame_bytes
    ysz, csz = EVAL_W * EVAL_H, EVAL_W * EVAL_H // 4
    with open(dst_raw, "wb") as fout:
        for i in range(n):
            off = i * frame_bytes
            y = np.array(data[off:off + ysz]).reshape(EVAL_H, EVAL_W)
            u = np.array(data[off + ysz:off + ysz + csz]).reshape(EVAL_H // 2, EVAL_W // 2)
            v = np.array(data[off + ysz + csz:off + ysz + 2 * csz]).reshape(EVAL_H // 2, EVAL_W // 2)
            rgb = upsample_to_camera(yuv_planes_to_rgb(y, u, v))
            fout.write(rgb.numpy().tobytes())
    return n


def main():
    if len(sys.argv) != 4:
        sys.exit("usage: inflate.py <archive_dir> <inflated_dir> <video_names_file>")
    archive_dir, inflated_dir, names_file = map(Path, sys.argv[1:4])
    inflated_dir.mkdir(parents=True, exist_ok=True)
    coolchic = Path(os.environ.get("COOLCHIC_REPO", str(Path.home() / "Cool-Chic"))).expanduser()

    bitstream = archive_dir / "bitstream.cool"
    if not bitstream.is_file():
        sys.exit(f"ERROR: {bitstream} not found (archive.zip must contain bitstream.cool)")

    names = [ln.strip() for ln in names_file.read_text().splitlines() if ln.strip()]
    # Single-video submission: one bitstream -> the first (only) name.
    for name in names:
        base = name.rsplit(".", 1)[0]
        recon_yuv = inflated_dir / f"recon_{EVAL_W}x{EVAL_H}_20p_yuv420_8b.yuv"
        decode_bitstream(bitstream, recon_yuv, coolchic)
        n = yuv_to_raw(recon_yuv, inflated_dir / f"{base}.raw")
        recon_yuv.unlink(missing_ok=True)
        print(f"inflated {base}.raw ({n} frames)", flush=True)


if __name__ == "__main__":
    main()
