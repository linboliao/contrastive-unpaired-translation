cd ../
export PYTHONPATH=.:$PYTHONPATH
#python train.py --name M1 --gpu_ids 1 --dataroot /NAS2/Data1/lbliao/Data/MSI/筛选后数据/M1 --n_epochs 50 --n_epochs_decay 50 --serial_batches --load_size 512 --crop_size 512
#python train.py --name M2 --gpu_ids 1 --dataroot /NAS2/Data1/lbliao/Data/MSI/筛选后数据/M2 --n_epochs 50 --n_epochs_decay 50 --serial_batches --load_size 512 --crop_size 512
#python train.py --name M6 --gpu_ids 1 --dataroot /NAS2/Data1/lbliao/Data/MSI/筛选后数据/M6 --n_epochs 50 --n_epochs_decay 50 --serial_batches --load_size 512 --crop_size 512
#python train.py --name CD34 --gpu_ids 1 --dataroot /NAS145/liaolinbo/Data/免疫治疗省肿瘤/60例胃癌/人工配准/dataset --n_epochs 50 --n_epochs_decay 50 --serial_batches --load_size 512 --crop_size 512 --continue_train
# baseline: 原生 CUT，无 DAB loss
python train.py --name ERG --gpu_ids 2 --dataroot /NAS145/liaolinbo/Data/免疫治疗省肿瘤/王建超第二批/配准/pair_0_2048/ERG_QC/CUT_105 --n_epochs 50 --n_epochs_decay 50 --serial_batches --load_size 512 --crop_size 512 --no_flip #--continue_train
# lambda_DAB=2.0，ambiguous 权重默认 0.3
python train.py --name ERG2 --gpu_ids 3 --dataroot /NAS145/liaolinbo/Data/免疫治疗省肿瘤/王建超第二批/配准/pair_0_2048/ERG_QC/CUT_105 --n_epochs 50 --n_epochs_decay 50 --serial_batches --load_size 512 --crop_size 512 --no_flip --lambda_DAB 2.0 #--continue_train
# lambda_DAB=5.0，DAB loss 权重加大的消融
python train.py --name ERG_dab_l5 --gpu_ids 0 --dataroot /NAS145/liaolinbo/Data/免疫治疗省肿瘤/王建超第二批/配准/pair_0_2048/ERG_QC/CUT_105 --n_epochs 50 --n_epochs_decay 50 --serial_batches --load_size 512 --crop_size 512 --no_flip --lambda_DAB 5.0
# lambda_DAB=2.0 + dab_ambiguous_weight=0.1，降低配准不确定样本权重的消融
python train.py --name ERG_dab_l2_ambig01 --gpu_ids 1 --dataroot /NAS145/liaolinbo/Data/免疫治疗省肿瘤/王建超第二批/配准/pair_0_2048/ERG_QC/CUT_105 --n_epochs 50 --n_epochs_decay 50 --serial_batches --load_size 512 --crop_size 512 --no_flip --lambda_DAB 2.0 --dab_ambiguous_weight 0.1
