#!/usr/bin/env bash
# Inference for the ERG_v3 sweep on the full, clean CUT_2 test split (106
# reg_group=A pairs -- see scripts/build_cut2_dataset.py). --num_test 106
# covers all of it, so no sampling. fastcut/netG6 already tested separately.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.:$PYTHONPATH

PY=/home/lbliao/anaconda3/envs/pix2pixHD/bin/python
DATAROOT="/NAS145/linboliao/Data/福建省肿瘤/免疫治疗/王建超第二批/配准/datasets/cut2"
COMMON="--dataroot $DATAROOT --load_size 512 --crop_size 512 --no_flip --num_test 106 --eval"

run() {
  name=$1; gpu=$2; extra=$3
  $PY test.py --name "$name" --gpu_ids "$gpu" $COMMON $extra > "logs/test_${name}.out" 2>&1 &
  echo "launched test $name on gpu $gpu (pid $!)"
}

run ERG_v3_baseline          0 ""
run ERG_v3_dab_l1            1 ""
run ERG_v3_dab_l2            2 ""
run ERG_v3_dab_l5            3 ""
run ERG_v3_dab_l2_ambig01    4 ""
run ERG_v3_dab_l2_ambig1     5 ""
wait
