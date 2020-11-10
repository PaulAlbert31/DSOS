#Mixup only testmethod
CUDA_VISIBLE_DEVICES=0 python main.py --dataset webvision --epochs 100 --batch-size 64 --net inception --lr 0.02 --steps 50 80 --seed 1 --exp-name fulltrain_0.02 --mixup --entro




