# Software Name: Cool-Chic (comma challenge extension)
# SPDX-License-Identifier: BSD 3-Clause "New"
"""comma.ai challenge distortion: frozen SegNet/PoseNet judge losses.

Activated with --tune=comma. The distortion for a decoded frame is:

  odd display index (frame 1 of a judged pair, SegNet + PoseNet see it):
      COMMA_SEG_W * seg_margin_surrogate(SegNet(render), gt_labels)
    + COMMA_POSE_W * sqrt(10 * MSE(PoseNet(ref_even, render)[:6], gt_pose))
    + COMMA_ODD_MSE_W * mse(decoded, target)

  even display index (frame 0, PoseNet-only via its pair; partner not yet
  coded, so only a pixel anchor keeps it a usable reference):
      COMMA_EVEN_MSE_W * mse(decoded, target)

The render chain replicates the official evaluation path end to end and is
fully differentiable: YUV420[0,1] -> RGB[0,255] (BT.601 limited, bilinear
chroma) -> bicubic upsample to 874x1164 -> clamp + straight-through round ->
DistortionNet preprocessing (with the same differentiable rgb_to_yuv6 patch
hnerv_muon uses: the challenge's own version is @torch.no_grad and severs
pose gradients).

Environment:
  COMMA_CHALLENGE_ROOT  challenge repo root (modules.py, models/) [required]
  COMMA_TARGETS_PT      targets file from prep_targets.py         [required]
  COMMA_SEG_W           default 100.0
  COMMA_POSE_W          default 1.0
  COMMA_EVEN_MSE_W      default 1.0
  COMMA_ODD_MSE_W       default 0.0
  COMMA_SEG_TAU         default 0.3

Frame context is set once per encoded frame by
``coolchic.component.video.encode_one_frame`` via :func:`set_frame_context`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from coolchic.io.format.yuv import DictTensorYUV

CAMERA_H, CAMERA_W = 874, 1164

_SEG_W = float(os.environ.get("COMMA_SEG_W", "100.0"))
_POSE_W = float(os.environ.get("COMMA_POSE_W", "1.0"))
_EVEN_MSE_W = float(os.environ.get("COMMA_EVEN_MSE_W", "1.0"))
_ODD_MSE_W = float(os.environ.get("COMMA_ODD_MSE_W", "0.0"))
_SEG_TAU = float(os.environ.get("COMMA_SEG_TAU", "0.3"))


class _Ctx:
    distortion_net = None
    seg_targets: Optional[Tensor] = None   # (n_pairs, 384, 512) long
    pose_targets: Optional[Tensor] = None  # (n_pairs, 6) float
    display_idx: int = -1
    ref_rgb: Optional[Tensor] = None       # (874, 1164, 3) float 0-255, no grad
    warned_no_ref: bool = False
    warned_no_ctx: bool = False


_ctx = _Ctx()


def _challenge_root() -> Path:
    root = os.environ.get("COMMA_CHALLENGE_ROOT")
    if not root:
        raise RuntimeError(
            "--tune=comma requires COMMA_CHALLENGE_ROOT pointing at the "
            "comma_video_compression_challenge repo")
    return Path(root).expanduser().resolve()


def _rgb_to_yuv6_differentiable(rgb_chw: Tensor) -> Tensor:
    """Differentiable copy of frame_utils.rgb_to_yuv6 (theirs is @no_grad)."""
    H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
    H2, W2 = H // 2, W // 2
    rgb = rgb_chw[..., :, : 2 * H2, : 2 * W2]
    R, G, B = rgb[..., 0, :, :], rgb[..., 1, :, :], rgb[..., 2, :, :]
    Y = (R * 0.299 + G * 0.587 + B * 0.114).clamp(0.0, 255.0)
    U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
    V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)
    U_sub = (U[..., 0::2, 0::2] + U[..., 1::2, 0::2]
             + U[..., 0::2, 1::2] + U[..., 1::2, 1::2]) * 0.25
    V_sub = (V[..., 0::2, 0::2] + V[..., 1::2, 0::2]
             + V[..., 0::2, 1::2] + V[..., 1::2, 1::2]) * 0.25
    return torch.stack([Y[..., 0::2, 0::2], Y[..., 1::2, 0::2],
                        Y[..., 0::2, 1::2], Y[..., 1::2, 1::2],
                        U_sub, V_sub], dim=-3)


def _judge_device() -> torch.device:
    # Judges are pinned to a single device for the whole encode. Training runs
    # on CUDA; the only calls on another device are cc_encode's cosmetic CPU
    # logging passes (model moved to CPU for the bitstream), handled by the
    # device-mismatch fallback in compute_comma_distortion.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_judges(device: torch.device):
    if _ctx.distortion_net is not None:
        return
    dev = _judge_device()
    root = _challenge_root()
    sys.path.insert(0, str(root))
    import frame_utils   # noqa: E402  (challenge repo)
    import modules       # noqa: E402
    frame_utils.rgb_to_yuv6 = _rgb_to_yuv6_differentiable
    modules.rgb_to_yuv6 = _rgb_to_yuv6_differentiable

    net = modules.DistortionNet().eval().to(dev)
    net.load_state_dicts(modules.posenet_sd_path, modules.segnet_sd_path, dev)
    for p in net.parameters():
        p.requires_grad = False
    _ctx.distortion_net = net

    targets_path = os.environ.get("COMMA_TARGETS_PT")
    if not targets_path:
        raise RuntimeError("--tune=comma requires COMMA_TARGETS_PT "
                           "(build it with prep_targets.py)")
    blob = torch.load(targets_path, map_location=dev)
    _ctx.seg_targets = blob["seg"].long().to(dev)
    _ctx.pose_targets = blob["pose"].float().to(dev)
    print(f"[comma metric] judges loaded on {dev}; targets for "
          f"{_ctx.seg_targets.shape[0]} pairs", flush=True)


def _yuv420_to_rgb255(yuv: DictTensorYUV) -> Tensor:
    """DictTensorYUV [0,1] (1,1,h,w planes) -> (1,3,h,w) RGB float in [0,255].

    Matches frame_utils.yuv420_to_rgb: BT.601 limited range, bilinear chroma
    upsampling. Differentiable (no round here).
    """
    y = yuv["y"] * 255.0
    u = yuv["u"] * 255.0
    v = yuv["v"] * 255.0
    h, w = y.shape[-2], y.shape[-1]
    u_up = F.interpolate(u, size=(h, w), mode="bilinear", align_corners=False)
    v_up = F.interpolate(v, size=(h, w), mode="bilinear", align_corners=False)
    yf = (y - 16.0) * (255.0 / 219.0)
    uf = (u_up - 128.0) * (255.0 / 224.0)
    vf = (v_up - 128.0) * (255.0 / 224.0)
    r = (yf + 1.402 * vf).clamp(0, 255)
    g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255)
    b = (yf + 1.772 * uf).clamp(0, 255)
    return torch.cat([r, g, b], dim=1)


def _to_camera_frame(yuv: DictTensorYUV, ste_round: bool) -> Tensor:
    """Decoded YUV420 at coding res -> (874, 1164, 3) float 0-255 camera frame,
    through the exact inflate chain (bicubic upsample, clamp, round)."""
    rgb = _yuv420_to_rgb255(yuv)
    if rgb.shape[-2:] != (CAMERA_H, CAMERA_W):
        rgb = F.interpolate(rgb, size=(CAMERA_H, CAMERA_W), mode="bicubic",
                            align_corners=False)
    rgb = rgb.clamp(0.0, 255.0)
    if ste_round:
        rgb = rgb + (rgb.round() - rgb).detach()
    else:
        rgb = rgb.round()
    return rgb.squeeze(0).permute(1, 2, 0)  # (H, W, 3)


def set_frame_context(frame, device: torch.device) -> None:
    """Called once per frame by encode_one_frame when tune is comma."""
    _load_judges(device)
    _ctx.display_idx = int(frame.display_order)
    _ctx.ref_rgb = None
    if _ctx.display_idx % 2 == 1:
        partner = _ctx.display_idx - 1
        for idx_ref, ref_data in zip(frame.index_references, frame.refs_data):
            if int(idx_ref) == partner:
                with torch.no_grad():
                    _ctx.ref_rgb = _to_camera_frame(
                        {k: v.to(device) for k, v in ref_data.data.items()},
                        ste_round=False,
                    )
                break
        if _ctx.ref_rgb is None and not _Ctx.warned_no_ref:
            _Ctx.warned_no_ref = True
            print(f"[comma metric] WARNING: odd frame {_ctx.display_idx} has no "
                  f"decoded even partner among refs {frame.index_references}; "
                  f"pose loss disabled for such frames (use --struct ippp)",
                  flush=True)
    print(f"[comma metric] frame {_ctx.display_idx} "
          f"({'odd/judged' if _ctx.display_idx % 2 else 'even/anchor'}, "
          f"pair ref {'yes' if _ctx.ref_rgb is not None else 'no'})", flush=True)


def _mse_yuv(decoded: DictTensorYUV, target: DictTensorYUV) -> Tensor:
    num = torch.zeros((), device=decoded["y"].device)
    den = 0
    for k in ("y", "u", "v"):
        num = num + F.mse_loss(decoded[k], target[k], reduction="sum")
        den += target[k].numel()
    return num / den


def compute_comma_distortion(
    decoded_image: Union[Tensor, DictTensorYUV],
    target_image: Union[Tensor, DictTensorYUV],
) -> Tensor:
    if isinstance(decoded_image, Tensor):
        raise ValueError("--tune=comma expects YUV420 input (DictTensorYUV)")
    _load_judges(decoded_image["y"].device)
    net = _ctx.distortion_net
    judge_device = next(net.parameters()).device
    caller_device = decoded_image["y"].device

    # loss.py builds final_dist on the CALLER's device and adds our return
    # value to it, so we MUST return a scalar on caller_device. The judges are
    # pinned to the training device (CUDA). Two cases arrive on another device,
    # both cosmetic (cc_encode moves the frame_encoder to CPU for the bitstream
    # then runs one RD-logging loss_function): a resumed frame (no context) and
    # the post-training log line. In both, skip the judges and report MSE on
    # the caller's device so the encode doesn't crash and no cross-device add
    # occurs. The real training path always has caller_device == judge_device.
    d = _ctx.display_idx
    if caller_device != judge_device or d < 0:
        if not _Ctx.warned_no_ctx:
            _Ctx.warned_no_ctx = True
            print("[comma metric] cosmetic logging/resume pass "
                  f"(caller {caller_device}, judges {judge_device}, ctx {d}); "
                  "reporting MSE for this log line only", flush=True)
        return _mse_yuv(decoded_image, target_image)

    anchor = _mse_yuv(decoded_image, target_image)

    if d % 2 == 0:
        return _EVEN_MSE_W * anchor

    dist = _ODD_MSE_W * anchor
    pair = d // 2
    cur = _to_camera_frame(decoded_image, ste_round=True)  # (H,W,3) 0-255

    # SegNet: judged on the last (odd) frame of the pair only.
    seg_in = net.segnet.preprocess_input(
        cur.permute(2, 0, 1).unsqueeze(0).unsqueeze(1).float())
    seg_logits = net.segnet(seg_in)
    tgt = _ctx.seg_targets[pair].unsqueeze(0)
    target_logits = seg_logits.gather(1, tgt.unsqueeze(1))
    masked = seg_logits.clone()
    masked.scatter_(1, tgt.unsqueeze(1), -1e9)
    margin = target_logits - masked.max(dim=1, keepdim=True)[0]
    seg_l = (_SEG_TAU * F.softplus(-margin / _SEG_TAU)).mean()
    dist = dist + _SEG_W * seg_l

    # PoseNet: needs the decoded even partner (frozen) + current frame.
    if _ctx.ref_rgb is not None:
        pair_t = torch.stack([_ctx.ref_rgb, cur]).unsqueeze(0)  # (1,2,H,W,3)
        pose_in = net.posenet.preprocess_input(
            pair_t.permute(0, 1, 4, 2, 3).float())
        pose_out = net.posenet(pose_in)
        pose_mse = F.mse_loss(pose_out["pose"][:, :6],
                              _ctx.pose_targets[pair].unsqueeze(0))
        dist = dist + _POSE_W * torch.sqrt(10.0 * pose_mse + 1e-12)

    return dist
