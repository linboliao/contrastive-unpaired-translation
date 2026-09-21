#!/usr/bin/env bash
# ERG v5: single-variable test of stain normalization + HED augmentation,
# against ERG_v4_dab_l1 as the control. Same split (cut3), same merged QC
# table, same lambda_DAB -- only the colour handling differs, so a difference
# in object-level vessel F1 is attributable.
#
# Method choice is empirical, not conventional: on this data Reinhard cuts
# cross-slide tissue-colour variance 64% while Macenko cuts 0% (it slightly
# increases it). See data/stain_transforms.ReinhardNormalizer for the numbers.
# hed_aug 0.05 is the visually calibrated value; 0.20+ leaves the H&E manifold.
#
# Normalization is applied to A (HE input) only by default. Normalizing B would
# also rescale the DAB optical density that the paired loss and every
# downstream vessel metric are computed from -- worth trying later, but it is a
# second variable, so it is not in this run.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.:$PYTHONPATH

PY=/home/lbliao/anaconda3/envs/pix2pixHD/bin/python
BASE="/NAS145/linboliao/Data/福建省肿瘤/免疫治疗/王建超第二批/配准"
DATAROOT="$BASE/datasets/cut3"
DABTSV="$BASE/qc/pair_scores_all.tsv"
# Reference is fitted from the training split only -- fitting it on anything
# that includes val/test slides would leak their colour statistics.
STAINREF="$DATAROOT/trainA"
COMMON="--dataroot $DATAROOT --dab_scores_tsv $DABTSV --n_epochs 50 --n_epochs_decay 50 --serial_batches --load_size 512 --crop_size 512 --no_flip --lambda_DAB 1.0"

mkdir -p logs

run() {
  name=$1; gpu=$2; extra=$3
  nohup $PY train.py --name "$name" --gpu_ids "$gpu" $COMMON $extra \
    > "logs/${name}.out" 2>&1 &
  echo "launched $name on gpu $gpu (pid $!)"
}

run ERG_v5_norm        4 "--stain_norm reinhard --stain_ref_A $STAINREF"
run ERG_v5_hed         5 "--hed_aug 0.05 --hed_aug_bias 0.05"
run ERG_v5_norm_hed    6 "--stain_norm reinhard --stain_ref_A $STAINREF --hed_aug 0.05 --hed_aug_bias 0.05"
run ERG_v5_norm_hed_ab 7 "--stain_norm reinhard --stain_norm_domains AB --stain_ref_A $STAINREF --stain_ref_B $DATAROOT/trainB --hed_aug 0.05 --hed_aug_bias 0.05"
