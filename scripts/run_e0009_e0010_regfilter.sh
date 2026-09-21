#!/usr/bin/env bash
# e0009/e0010: registration hard-filter ablation. Control = ERG_v4_dab_l1
# (e0002, lambda_DAB=1, continuous reg_score soft-weighting). This tests
# --dab_min_reg_score (new flag, models/stain_utils.py + models/cut_model.py):
# pairs with reg_score below the threshold get paired-DAB weight forced to
# 0.0 (hard-excluded from that loss term only -- they still participate in
# the unpaired GAN/PatchNCE losses since --serial_batches only pairs indices
# for the DAB term, image population is unchanged).
#
# Thresholds are cut3 train reg_score quantiles (n=3155, range 0.300-0.519,
# median 0.382): p50=0.382 (drop bottom half) and p75=0.416 (keep only the
# best quartile). A fixed 0.50 "project qualifying line" would leave almost
# nothing -- cut3's max reg_score is 0.519.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.:$PYTHONPATH

PY=/home/lbliao/anaconda3/envs/pix2pixHD/bin/python
BASE="/NAS145/linboliao/Data/福建省肿瘤/免疫治疗/王建超第二批/配准"
DATAROOT="$BASE/datasets/cut3"
DABTSV="$BASE/qc/pair_scores_all.tsv"
COMMON="--dataroot $DATAROOT --dab_scores_tsv $DABTSV --n_epochs 50 --n_epochs_decay 50 --serial_batches --load_size 512 --crop_size 512 --no_flip --lambda_DAB 1.0"

mkdir -p logs

run() {
  name=$1; gpu=$2; extra=$3
  nohup $PY train.py --name "$name" --gpu_ids "$gpu" $COMMON $extra \
    > "logs/${name}.out" 2>&1 &
  echo "launched $name on gpu $gpu (pid $!)"
}

run e0009 0 "--dab_min_reg_score 0.382"
run e0010 1 "--dab_min_reg_score 0.416"
