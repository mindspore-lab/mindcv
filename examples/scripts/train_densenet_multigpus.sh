export CUDA_VISIBLE_DEVICES=0,1,2,3
mpirun --allow-run-as-root -n 4 python train.py --distribute --model=densenet121 --pretrained --epoch_size=5 --dataset=cifar10 --dataset_download
#mpirun --allow-run-as-root -n 4 python train.py --distribute --model=densenet121 --pretrained --epoch_size=5 --dataset=cifar10 --data_dir=./datasets/cifar/cifar-10-batches-bin
