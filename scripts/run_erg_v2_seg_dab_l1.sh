#!/usr/bin/env bash
# Bolts a nucleus segmentation head onto CUT (models/cut_seg_model.py),
# warm-started from ERG_v2_dab_l1 -- the closest-calibrated run in the
# ERG_v2 sweep (r_median=1.10, see qc/erg_v2_sweep_scores.tsv). netG/netF/netD
# load ERG_v2_dab_l1's weights as-is (--pretrained_name); netSeg starts fresh
# (see CUTSegModel.load_networks). GAN/DAB losses are turned down, not
# removed, so capacity shifts toward the Dice+focal segmentation loss.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.:$PYTHONPATH

PY=/home/lbliao/anaconda3/envs/pix2pixHD/bin/python
DATAROOT="/NAS145/linboliao/Data/福建省肿瘤/免疫治疗/王建超第二批/配准/ERG_QC/CUT_105"
DABTSV="/NAS145/linboliao/Data/福建省肿瘤/免疫治疗/王建超第二批/配准/ERG_QC/pair_scores.tsv"
GPU=${1:-0}

mkdir -p logs

nohup $PY train.py \
  --name ERG_v2_dab_l1_seg \
  --model cut_seg \
  --gpu_ids "$GPU" \
  --continue_train --pretrained_name ERG_v2_dab_l1 --epoch latest \
  --dataroot "$DATAROOT" --dab_scores_tsv "$DABTSV" \
  --lambda_GAN 0.3 --lambda_DAB 0.3 \
  --lambda_seg_dice 1.0 --lambda_seg_focal 1.0 \
  --n_epochs 50 --n_epochs_decay 50 \
  --serial_batches --load_size 512 --crop_size 512 --no_flip \
  > "logs/ERG_v2_dab_l1_seg.out" 2>&1 &
echo "launched ERG_v2_dab_l1_seg on gpu $GPU (pid $!)"
