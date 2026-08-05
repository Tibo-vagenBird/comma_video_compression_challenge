#!/usr/bin/env python
"""Post-hoc: can per-channel Gaussian entropy coding beat the shipped coder on
the REAL trained #101/#95 hnerv decoder weights?

Coding-only test (same INT8 quantization the submission ships): compares the
Shannon cost of several memoryless entropy models against the actually-shipped
161,104-byte ctx-coded size. Isolates the CODER; the quantization is fixed.

Run from the repo root (needs torch, numpy; brotli optional):
  python experiments/weight_entropy_probe.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "submissions" / "rhnerv_latent_polish"
sys.path.insert(0, str(SUB))

import codec  # rhnerv_latent_polish/codec.py  (brings HNeRVDecoder + constants)

SHIPPED_CTX_BYTES = 161_104   # decoder section actually shipped in archive.zip
RAW_INT8_BYTES = 229_014      # decoder_streams.bin: raw per-tensor int8 + fp16 scale


def load_int8_tensors():
    """Replicate decode_decoder_compact but keep the int8 codes q per tensor."""
    raw = (SUB / "encoder" / "decoder_streams.bin").read_bytes()
    probe = codec.HNeRVDecoder(latent_dim=codec.LATENT_DIM,
                               base_channels=codec.BASE_CHANNELS,
                               eval_size=codec.EVAL_SIZE)
    items = list(probe.state_dict().items())
    pos = 0
    out = []
    for idx in codec.DECODER_STORAGE_ORDER:
        name, tensor = items[idx]
        shape = tuple(tensor.shape)
        numel = int(tensor.numel())
        zz = np.frombuffer(raw, dtype=np.uint8, count=numel, offset=pos)
        pos += numel
        scale = float(np.frombuffer(raw, dtype=np.float16, count=1, offset=pos)[0])
        pos += 2
        q = codec.decode_mapped_u8(zz, codec.DECODER_BYTE_MAPS.get(idx, "zig"))
        if len(shape) == 4:
            stored_shape = tuple(shape[i] for i in codec.CONV4_STORAGE_PERMS[idx])
            q = np.transpose(q.reshape(stored_shape),
                             codec.CONV4_INVERSE_PERMS[idx]).copy()
        else:
            q = q.reshape(shape)
        out.append((name, q.astype(np.int32), scale, shape))
    return out


def order0_bits(vals):
    """Shannon bits for iid symbols under their empirical distribution."""
    v, c = np.unique(vals, return_counts=True)
    p = c / c.sum()
    return float(-(c * np.log2(p)).sum())  # = N * H


def gaussian_bits(vals):
    """Bits to code int symbols under a fitted discretized Gaussian N(mu,sigma)."""
    from math import erf, sqrt
    mu, sigma = float(vals.mean()), float(vals.std())
    if sigma < 1e-6:
        return 0.0  # constant channel: free given side info
    lo, hi = int(vals.min()), int(vals.max())
    grid = np.arange(lo, hi + 1)
    cdf = 0.5 * (1 + np.vectorize(lambda x: erf(x))((grid + 0.5 - mu) / (sigma * sqrt(2))))
    cdf_lo = 0.5 * (1 + np.vectorize(lambda x: erf(x))((grid - 0.5 - mu) / (sigma * sqrt(2))))
    prob = np.clip(cdf - cdf_lo, 2**-16, 1.0)
    logp = {int(g): -np.log2(prob[i]) for i, g in enumerate(grid)}
    return float(sum(logp[int(x)] for x in vals))


def main():
    tensors = load_int8_tensors()
    n_params = sum(q.size for _, q, _, _ in tensors)
    print(f"decoder: {len(tensors)} tensors, {n_params:,} params\n")

    per_tensor = per_ch_ord0 = per_ch_gauss = 0.0
    side_pt = side_pc = 0
    print(f"{'tensor':<26}{'shape':<20}{'/tensor':>9}{'/ch-ord0':>10}{'/ch-gauss':>11}")
    for name, q, scale, shape in tensors:
        n = q.size
        bt = order0_bits(q.ravel())
        per_tensor += bt
        side_pt += 2  # one fp16 scale per tensor

        if q.ndim >= 2:                       # per-output-channel (dim 0)
            chans = q.reshape(q.shape[0], -1)
            bo = sum(order0_bits(chans[c]) for c in range(chans.shape[0]))
            bg = sum(gaussian_bits(chans[c]) for c in range(chans.shape[0]))
            side_pc += chans.shape[0] * 4     # fp16 mean + fp16 scale per channel
        else:
            bo = bt
            bg = gaussian_bits(q.ravel())
            side_pc += 4
        per_ch_ord0 += bo
        per_ch_gauss += bg
        if n >= 4000:
            print(f"{name:<26}{str(tuple(shape)):<20}"
                  f"{bt/8:>9,.0f}{bo/8:>10,.0f}{bg/8:>11,.0f}")

    def tot(bits, side):
        return bits / 8 + side

    print("\n=== coding-only comparison (same INT8 quantization) ===")
    print(f"  shipped ctx coder (actual)      : {SHIPPED_CTX_BYTES:>8,} B")
    print(f"  raw int8 + fp16 scale (uncoded) : {RAW_INT8_BYTES:>8,} B")
    print(f"  per-TENSOR order-0 entropy       : {tot(per_tensor, side_pt):>8,.0f} B"
          f"  (adaptive per-tensor floor ~ ctx coder)")
    print(f"  per-CHANNEL order-0 entropy      : {tot(per_ch_ord0, side_pc):>8,.0f} B"
          f"  (+{side_pc:,} B side info)")
    print(f"  per-CHANNEL Gaussian (NVRC-style): {tot(per_ch_gauss, side_pc):>8,.0f} B"
          f"  (+{side_pc:,} B side info)")

    try:
        import brotli
        allq = np.concatenate([
            np.where(q.ravel() >= 0, 2 * q.ravel(), -2 * q.ravel() - 1).astype(np.uint8)
            for _, q, _, _ in tensors])
        bro = len(brotli.compress(allq.tobytes(), quality=11)) + side_pt
        print(f"  hnerv method (zigzag+brotli)     : {bro:>8,} B")
    except ImportError:
        print("  (brotli not installed - skipping hnerv-method baseline)")

    best = tot(min(per_ch_ord0, per_ch_gauss), side_pc)
    delta = SHIPPED_CTX_BYTES - best
    print(f"\n  best new scheme vs shipped ctx: {delta:+,.0f} B "
          f"({100*delta/SHIPPED_CTX_BYTES:+.1f}% ; score {25*delta/37_545_489:+.5f})")


if __name__ == "__main__":
    main()
