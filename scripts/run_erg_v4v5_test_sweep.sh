#!/usr/bin/env bash
# Historical inference launcher for the ERG_v4/v5 sweep.
# It intentionally retains the original --num_test 522 setting for provenance.
# Current formal evaluation must use the cut3 manifest whitelist (521 valid
# test stems); do not treat this launcher as the current evaluation protocol.
# Inference for the ERG_v4/v5 sweep on the slide-disjoint cut3 test split
# (522 tiles, 3 held-out slides -- see datasets/cut3/split_manifest.tsv).
#
# v5_norm_hed_ab is deliberately run with --stain_norm_domains A (NOT the
# training-time AB) so the real_B saved to results/ stays the raw,
# unnormalized IHC reference -- comparable to the other 7 groups. Domain-A
# normalization is still applied so the generator sees the input
# distribution it was trained on; stain_norm has no isTrain gate, so B would
# otherwise come out normalized at test time too and quietly break the
# target-level comparison (see PROJECT_GOVERNANCE.md / CLAUDE.md).
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.:$PYTHONPATH

PY=/home/lbliao/anaconda3/envs/pix2pixHD/bin/python
BASE="/NAS145/linboliao/Data/福建省肿瘤/免疫治疗/王建超第二批/配准"
DATAROOT="$BASE/datasets/cut3"
STAINREF="$DATAROOT/trainA"
COMMON="--dataroot $DATAROOT --load_size 512 --crop_size 512 --no_flip --num_test 522 --eval"

mkdir -p logs

run() {
  name=$1; gpu=$2; extra=$3
  $PY test.py --name "$name" --gpu_ids "$gpu" $COMMON $extra > "logs/test_${name}.out" 2>&1 &
  echo "launched test $name on gpu $gpu (pid $!)"
}

run ERG_v4_baseline     0 ""
run ERG_v4_dab_l1       1 ""
run ERG_v4_dab_l5       2 ""
run ERG_v4_fastcut      4 ""
run ERG_v5_norm         5 "--stain_norm reinhard --stain_ref_A $STAINREF"
run ERG_v5_hed          6 ""
run ERG_v5_norm_hed     7 "--stain_norm reinhard --stain_ref_A $STAINREF"
wait -n
run ERG_v5_norm_hed_ab  0 "--stain_norm reinhard --stain_norm_domains A --stain_ref_A $STAINREF"
wait
