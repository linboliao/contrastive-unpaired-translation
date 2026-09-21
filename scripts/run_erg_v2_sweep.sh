#!/usr/bin/env bash
# ERG v2 sweep: 8 parallel single-GPU CUT training runs on the rebuilt
# ERG_QC/CUT_105 dataset (crop_1748 edge-fix + merged strong_A/weak_B).
# Ablates the paired DAB registration loss (lambda_DAB, dab_ambiguous_weight)
# plus CUT_mode / netG capacity. One job per GPU (each run uses ~10.4GB on a
# 12GB 3080 Ti at crop_size=512, so do not co-locate two runs on one GPU).
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.:$PYTHONPATH

PY=/home/lbliao/anaconda3/envs/pix2pixHD/bin/python
DATAROOT="/NAS145/linboliao/Data/福建省肿瘤/免疫治疗/王建超第二批/配准/ERG_QC/CUT_105"
DABTSV="/NAS145/linboliao/Data/福建省肿瘤/免疫治疗/王建超第二批/配准/ERG_QC/pair_scores.tsv"
COMMON="--dataroot $DATAROOT --dab_scores_tsv $DABTSV --n_epochs 50 --n_epochs_decay 50 --serial_batches --load_size 512 --crop_size 512 --no_flip"

mkdir -p logs

run() {
  name=$1; gpu=$2; extra=$3
  nohup $PY train.py --name "$name" --gpu_ids "$gpu" $COMMON $extra \
    > "logs/${name}.out" 2>&1 &
  echo "launched $name on gpu $gpu (pid $!)"
}

run ERG_v2_baseline          0 "--lambda_DAB 0"
run ERG_v2_dab_l1            1 "--lambda_DAB 1.0"
run ERG_v2_dab_l2            2 "--lambda_DAB 2.0"
run ERG_v2_dab_l5            3 "--lambda_DAB 5.0"
run ERG_v2_dab_l2_ambig01    4 "--lambda_DAB 2.0 --dab_ambiguous_weight 0.1"
run ERG_v2_dab_l2_ambig1     5 "--lambda_DAB 2.0 --dab_ambiguous_weight 1.0"
run ERG_v2_fastcut_dab_l2    6 "--CUT_mode FastCUT --lambda_DAB 2.0"
run ERG_v2_dab_l2_netG6      7 "--lambda_DAB 2.0 --netG resnet_6blocks"
