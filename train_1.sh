#Mixup only testmethod
#CUDA_VISIBLE_DEVICES=1 python main.py --dataset miniimagenet_preset --epochs 100 --batch-size 64 --net preresnet18 --lr 0.1 --steps 50 80 --seed 1 --exp-name ablastud_mix --mixup --noise-ratio 0.3

CUDA_VISIBLE_DEVICES=1 python main.py --dataset miniimagenet_preset --epochs 100 --batch-size 64 --net preresnet18 --lr 0.1 --steps 50 80 --seed 1 --exp-name ablastud_softmix --mixup --noise-ratio 0.3

CUDA_VISIBLE_DEVICES=1 python main.py --dataset miniimagenet_preset --epochs 100 --batch-size 64 --net preresnet18 --lr 0.1 --steps 50 80 --seed 1 --exp-name ablastud_softmixentro --mixup --entro --noise-ratio 0.3



