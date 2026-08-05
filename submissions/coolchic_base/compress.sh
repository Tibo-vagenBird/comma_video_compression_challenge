#!/usr/bin/env bash
# Reproduce the coolchic_base encode (reference; produces bitstream.cool).
#
# This is NOT run by the challenge harness — it documents how the shipped
# archive was produced. It needs the Cool-Chic 5.0.1 stack WITH the comma
# judge-loss modifications applied (see coolchic_mods/), and the challenge
# repo (for the frozen SegNet/PoseNet + videos/0.mkv).
#
# Steps (see encode/ for the scripts, all as of the 1.33 version):
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CCREPO="${COOLCHIC_REPO:-$HOME/Cool-Chic}"
CHALLENGE_ROOT="$(cd "$HERE/../.." && pwd)"

cat <<EOF
coolchic_base reproduction (1.33 config: lambda=40, IPPP, eval 512x384, --tune=comma)

1) Apply the comma modifications to a Cool-Chic 5.0.1 checkout:
     cp coolchic_mods/comma.py  \$COOLCHIC_REPO/coolchic/training/metrics/comma.py
     cd \$COOLCHIC_REPO && git apply $HERE/coolchic_mods/integration.patch
   (Or check out Cool-Chic at commit f59019e, which already contains them.)

2) In the *comma-compress* env (challenge repo), prepare inputs:
     python $HERE/encode/prep_yuv.sh    32 eval        # -> data/video0_512x384_...yuv
     python $HERE/encode/prep_targets.py --n-frames 32 # -> data/targets_n32.pt

3) In the *coolchic* env, encode (this is the run that scored 1.33):
     COMMA_CHALLENGE_ROOT=$CHALLENGE_ROOT \\
     bash $HERE/encode/run_one.sh \$COOLCHIC_REPO 40 32 <gpu> ippp eval comma
   -> runs/lmbda_40_n32_ippp_eval_comma_r01/bitstream.cool

4) Package:
     zip -j archive.zip <run>/bitstream.cool

5) Score (comma-compress env):
     python $HERE/encode/score_coolchic.py --run <run>

Full submission: repeat step 2-3 with n_frames = 1200 (the whole video),
not 32.  The shipped archive.zip is the 32-frame proof-of-concept.
EOF
