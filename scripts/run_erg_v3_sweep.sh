#!/usr/bin/env bash
# ERG v3 sweep: the same 8 configurations as scripts/run_erg_v2_sweep.sh
# (see that file for what each ablates), rerun on CUT_2 -- the stratified
# dataset (scripts/build_cut2_dataset.py) that fixes the eval-methodology
# bug in v2: trainA/B = all reg_group B pairs (3919, weak registration, which
# CUT tolerates by design), testA/B = all reg_group A pairs (106, strict
# registration QC thresholds) so paired pixel metrics on the test set are
# actually trustworthy. v2 checkpoints are archived at
# checkpoints_archive/v1_mixed_split/ for reference/comparison.
# One job per GPU (~10.4GB on a 12GB 3080 Ti at crop_size=512).
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.:$PYTHONPATH

PY=/home/lbliao/anaconda3/envs/pix2pixHD/bin/python
DATAROOT="/NAS145/linboliao/Data/福建省肿瘤/免疫治疗/王建超第二批/配准/datasets/cut2"
DABTSV="/NAS145/linboliao/Data/福建省肿瘤/免疫治疗/王建超第二批/配准/qc/105/pair_scores.tsv"
COMMON="--dataroot $DATAROOT --dab_scores_tsv $DABTSV --n_epochs 50 --n_epochs_decay 50 --serial_batches --load_size 512 --crop_size 512 --no_flip"

mkdir -p logs

run() {
  name=$1; gpu=$2; extra=$3
  nohup $PY train.py --name "$name" --gpu_ids "$gpu" $COMMON $extra \
    > "logs/${name}.out" 2>&1 &
  echo "launched $name on gpu $gpu (pid $!)"
}

run ERG_v3_baseline          0 "--lambda_DAB 0"
run ERG_v3_dab_l1            1 "--lambda_DAB 1.0"
run ERG_v3_dab_l2            2 "--lambda_DAB 2.0"
run ERG_v3_dab_l5            3 "--lambda_DAB 5.0"
run ERG_v3_dab_l2_ambig01    4 "--lambda_DAB 2.0 --dab_ambiguous_weight 0.1"
run ERG_v3_dab_l2_ambig1     5 "--lambda_DAB 2.0 --dab_ambiguous_weight 1.0"
run ERG_v3_fastcut_dab_l2    6 "--CUT_mode FastCUT --lambda_DAB 2.0"
run ERG_v3_dab_l2_netG6      7 "--lambda_DAB 2.0 --netG resnet_6blocks"
