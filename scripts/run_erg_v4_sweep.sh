#!/usr/bin/env bash
# ERG v4: first training run on the slide-disjoint split (datasets/cut3).
#
# Why four configs instead of v3's eight: the v3 sweep varied lambda_DAB and
# dab_ambiguous_weight across eight jobs and could not separate them -- every
# SSIM landed in 0.494-0.521 against a within-config std of 0.062, and
# object-level vessel F1 came out 0.13-0.19 with per-tile count correlation
# near zero. Those knobs all scale the *same* paired DAB loss, whose
# supervision target is misaligned on ~77% of tiles, so re-sweeping them buys
# nothing. v4 keeps only the two ends of that axis, to re-establish a baseline
# on a split that is not leaking.
#
# IMPORTANT: --dab_scores_tsv points at the MERGED qc/pair_scores_all.tsv.
# cut3 pools both batches, and RegistrationWeightTable.get() silently returns
# weight 1.0 for any stem it cannot find -- with the old per-batch tsv, 1070 of
# the 3155 training tiles (33.9%, real reg_score median 0.390) would have been
# trained at maximum paired-loss confidence. Exactly backwards.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.:$PYTHONPATH

PY=/home/lbliao/anaconda3/envs/pix2pixHD/bin/python
BASE="/NAS145/linboliao/Data/福建省肿瘤/免疫治疗/王建超第二批/配准"
DATAROOT="$BASE/datasets/cut3"
DABTSV="$BASE/qc/pair_scores_all.tsv"
COMMON="--dataroot $DATAROOT --dab_scores_tsv $DABTSV --n_epochs 50 --n_epochs_decay 50 --serial_batches --load_size 512 --crop_size 512 --no_flip"

mkdir -p logs

run() {
  name=$1; gpu=$2; extra=$3
  nohup $PY train.py --name "$name" --gpu_ids "$gpu" $COMMON $extra \
    > "logs/${name}.out" 2>&1 &
  echo "launched $name on gpu $gpu (pid $!)"
}

run ERG_v4_dab_l1    0 "--lambda_DAB 1.0"                      # v3 目标级最优组，新基线
run ERG_v4_baseline  1 "--lambda_DAB 0"                        # 干净划分下配对 DAB 损失是否有用
run ERG_v4_dab_l5    2 "--lambda_DAB 5.0"                      # 该轴另一端
run ERG_v4_fastcut   3 "--CUT_mode FastCUT --lambda_DAB 1.0"   # 模式对比
