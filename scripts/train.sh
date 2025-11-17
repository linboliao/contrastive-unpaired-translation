cd ../
export PYTHONPATH=.:$PYTHONPATH
#python train.py --name M1 --gpu_ids 1 --dataroot /NAS2/Data1/lbliao/Data/MSI/筛选后数据/M1 --n_epochs 50 --n_epochs_decay 50 --serial_batches --load_size 512 --crop_size 512
#python train.py --name M2 --gpu_ids 1 --dataroot /NAS2/Data1/lbliao/Data/MSI/筛选后数据/M2 --n_epochs 50 --n_epochs_decay 50 --serial_batches --load_size 512 --crop_size 512
#python train.py --name M6 --gpu_ids 1 --dataroot /NAS2/Data1/lbliao/Data/MSI/筛选后数据/M6 --n_epochs 50 --n_epochs_decay 50 --serial_batches --load_size 512 --crop_size 512
python train.py --name P2 --gpu_ids 1 --dataroot /NAS2/Data1/lbliao/Data/MSI/筛选后数据/P2 --n_epochs 50 --n_epochs_decay 50 --serial_batches --load_size 512 --crop_size 512
