#!/usr/bin/env bash
# Runs inference (test.py) for all 9 finished ERG_v2 sweep checkpoints on the
# held-out test split (603 pairs), --num_test 200 each, one job per GPU so
# they all finish in under a minute. Results land in results/<name>/test_latest/.
# Feeds scripts/evaluate_erg_sweep.py, which needs identical --num_test across
# runs (filename-aligned testA/testB, so results are directly comparable).
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.:$PYTHONPATH

PY=/home/lbliao/anaconda3/envs/pix2pixHD/bin/python
DATAROOT="/NAS145/linboliao/Data/福建省肿瘤/免疫治疗/王建超第二批/配准/ERG_QC/CUT_105"
COMMON="--dataroot $DATAROOT --load_size 512 --crop_size 512 --no_flip --num_test 200 --eval"

run() {
  name=$1; gpu=$2; extra=$3
  $PY test.py --name "$name" --gpu_ids "$gpu" $COMMON $extra > "logs/test_${name}.out" 2>&1 &
  echo "launched test $name on gpu $gpu (pid $!)"
}

run ERG_v2_baseline          0 ""
run ERG_v2_dab_l1            1 ""
run ERG_v2_dab_l2            2 ""
run ERG_v2_dab_l5            3 ""
run ERG_v2_dab_l2_ambig01    4 ""
run ERG_v2_dab_l2_ambig1     5 ""
run ERG_v2_fastcut_dab_l2    6 "--CUT_mode FastCUT"
run ERG_v2_dab_l2_netG6      7 "--netG resnet_6blocks"
wait

run ERG_v2_dab_l2_noidt      0 ""
wait
