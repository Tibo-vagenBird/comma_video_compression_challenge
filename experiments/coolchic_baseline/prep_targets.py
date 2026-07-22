#!/usr/bin/env python
"""Precompute ground-truth SegNet/PoseNet targets for --tune=comma training.

Run in the comma-compress env from the challenge repo root:

  python experiments/coolchic_baseline/prep_targets.py --n-frames 32

Writes experiments/coolchic_baseline/data/targets_n<N>.pt containing
  seg:  (n_pairs, 384, 512) uint8   SegNet argmax labels of frame 1 of each pair
  pose: (n_pairs, 6) float32        PoseNet pose vector of each GT pair
computed through the exact harness decode path (PyAV + yuv420_to_rgb).
"""
import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

import av  # noqa: E402
from frame_utils import yuv420_to_rgb  # noqa: E402
from modules import DistortionNet, segnet_sd_path, posenet_sd_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=32)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    net = DistortionNet().eval().to(device)
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

    seg_targets, pose_targets = [], []
    container = av.open(str(ROOT / "videos" / "0.mkv"))
    prev = None
    with torch.inference_mode():
        for frame in container.decode(container.streams.video[0]):
            f = yuv420_to_rgb(frame)
            if prev is None:
                prev = f
                continue
            pair = torch.stack([prev, f]).unsqueeze(0).to(device)
            prev = None
            po, so = net(pair)
            seg_targets.append(so.argmax(dim=1).squeeze(0).to(torch.uint8).cpu())
            pose_targets.append(po["pose"][:, :6].float().squeeze(0).cpu())
            if 2 * len(seg_targets) >= args.n_frames:
                break
    container.close()

    out = HERE / "data" / f"targets_n{args.n_frames}.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save({"seg": torch.stack(seg_targets),
                "pose": torch.stack(pose_targets)}, out)
    print(f"wrote {out}: seg {tuple(torch.stack(seg_targets).shape)}, "
          f"pose {tuple(torch.stack(pose_targets).shape)}")


if __name__ == "__main__":
    main()
