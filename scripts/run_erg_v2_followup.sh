#!/usr/bin/env bash
# ERG v2 sweep follow-up: backfills GPU 6 after ERG_v2_fastcut_dab_l2 finished
# early. Isolates the nce_idt (identity-loss) dimension inside plain CUT mode,
# which the original 8-run sweep only ever toggled bundled together with
# FastCUT's other changes (flip_equivariance, lambda_NCE=10, epoch schedule).
# See scripts/run_erg_v2_sweep.sh for the rest of the sweep and conventions.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.:$PYTHONPATH

PY=/home/lbliao/anaconda3/envs/pix2pixHD/bin/python
DATAROOT="/NAS145/linboliao/Data/福建省肿瘤/免疫治疗/王建超第二批/配准/ERG_QC/CUT_105"
DABTSV="/NAS145/linboliao/Data/福建省肿瘤/免疫治疗/王建超第二批/配准/ERG_QC/pair_scores.tsv"
COMMON="--dataroot $DATAROOT --dab_scores_tsv $DABTSV --n_epochs 50 --n_epochs_decay 50 --serial_batches --load_size 512 --crop_size 512 --no_flip"

mkdir -p logs

name=ERG_v2_dab_l2_noidt
gpu=6
nohup $PY train.py --name "$name" --gpu_ids "$gpu" $COMMON --lambda_DAB 2.0 --nce_idt False \
  > "logs/${name}.out" 2>&1 &
echo "launched $name on gpu $gpu (pid $!)"
