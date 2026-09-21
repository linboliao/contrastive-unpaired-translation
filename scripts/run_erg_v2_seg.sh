#!/usr/bin/env bash
# Generic launcher for the CUTSegModel follow-up (models/cut_seg_model.py):
# bolts a nucleus segmentation head onto any finished ERG_v2 sweep checkpoint.
# Usage: run_erg_v2_seg.sh <pretrained_name> <gpu> [lambda_dab]
# e.g.:  run_erg_v2_seg.sh ERG_v2_dab_l2_ambig01 1
#        run_erg_v2_seg.sh ERG_v2_baseline 2 0   # keep DAB loss OFF, matching the base run
# New experiment is named <pretrained_name>_seg. netG/netF/netD warm-start
# from <pretrained_name>'s weights; netSeg starts fresh (see
# CUTSegModel.load_networks). See scripts/run_erg_v2_seg_dab_l1.sh (the
# original pilot run on ERG_v2_dab_l1) for background.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.:$PYTHONPATH

PY=/home/lbliao/anaconda3/envs/pix2pixHD/bin/python
DATAROOT="/NAS145/linboliao/Data/福建省肿瘤/免疫治疗/王建超第二批/配准/ERG_QC/CUT_105"
DABTSV="/NAS145/linboliao/Data/福建省肿瘤/免疫治疗/王建超第二批/配准/ERG_QC/pair_scores.tsv"
PRETRAINED=${1:?usage: run_erg_v2_seg.sh <pretrained_name> <gpu> [lambda_dab]}
GPU=${2:?usage: run_erg_v2_seg.sh <pretrained_name> <gpu> [lambda_dab]}
LAMBDA_DAB=${3:-0.3}
NAME="${PRETRAINED}_seg"

mkdir -p logs

nohup $PY train.py \
  --name "$NAME" \
  --model cut_seg \
  --gpu_ids "$GPU" \
  --continue_train --pretrained_name "$PRETRAINED" --epoch latest \
  --dataroot "$DATAROOT" --dab_scores_tsv "$DABTSV" \
  --lambda_GAN 0.3 --lambda_DAB "$LAMBDA_DAB" \
  --lambda_seg_dice 1.0 --lambda_seg_focal 1.0 \
  --n_epochs 50 --n_epochs_decay 50 \
  --serial_batches --load_size 512 --crop_size 512 --no_flip \
  > "logs/${NAME}.out" 2>&1 &
echo "launched $NAME on gpu $GPU (pid $!)"
