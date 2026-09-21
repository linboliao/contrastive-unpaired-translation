#!/usr/bin/env bash
# Inference for the 3 finished CUTSegModel runs on the same 200-pair held-out
# test split used by run_erg_v2_test_sweep.sh, for apples-to-apples
# comparison against the plain-CUT sweep. Saves real_A/fake_B/real_B (as
# before) plus seg_prob_vis/seg_target_vis (the segmentation head's output
# and its ground-truth mask) under results/<name>/test_latest/images/.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.:$PYTHONPATH

PY=/home/lbliao/anaconda3/envs/pix2pixHD/bin/python
DATAROOT="/NAS145/linboliao/Data/福建省肿瘤/免疫治疗/王建超第二批/配准/ERG_QC/CUT_105"
COMMON="--model cut_seg --dataroot $DATAROOT --load_size 512 --crop_size 512 --no_flip --num_test 200 --eval"

run() {
  name=$1; gpu=$2
  $PY test.py --name "$name" --gpu_ids "$gpu" $COMMON > "logs/test_${name}.out" 2>&1 &
  echo "launched test $name on gpu $gpu (pid $!)"
}

run ERG_v2_dab_l1_seg          0
run ERG_v2_dab_l2_ambig01_seg  1
run ERG_v2_baseline_seg        2
wait
