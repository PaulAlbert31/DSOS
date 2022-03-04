# Cross entropy baselines
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 100 --batch-size 64 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_20ood_20id_ce_IN32 --ood-ratio 0.2 --ind-ratio 0.2 > logs/logs_20o20i_ce.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 100 --batch-size 64 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_40ood_20id_ce_IN32 --ood-ratio 0.4 --ind-ratio 0.2 > logs/logs_40o20i_ce.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 100 --batch-size 64 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_60ood_20id_ce_IN32 --ood-ratio 0.6 --ind-ratio 0.2 > logs/logs_60o20i_ce.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 100 --batch-size 64 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_40ood_40id_ce_IN32 --ood-ratio 0.4 --ind-ratio 0.4 > logs/logs_40o40i_ce.txt

# Mixup baselines
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 100 --batch-size 64 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_20ood_20id_mixup_IN32 --ood-ratio 0.2 --ind-ratio 0.2 --mixup > logs/logs_20o20i_mixup.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 100 --batch-size 64 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_40ood_20id_mixup_IN32 --ood-ratio 0.4 --ind-ratio 0.2 --mixup > logs/logs_40o20i_mixup.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 100 --batch-size 64 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_60ood_20id_mixup_IN32 --ood-ratio 0.6 --ind-ratio 0.2 --mixup > logs/logs_60o20i_mixup.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 100 --batch-size 64 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_40ood_40id_mixup_IN32 --ood-ratio 0.4 --ind-ratio 0.4 --mixup > logs/logs_40o40i_mixup.txt

# Warmup with Mixup + entropy reg
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 50 --batch-size 64 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_20ood_20id_mixup_entro_IN32 --ood-ratio 0.2 --ind-ratio 0.2 --entro --mixup > logs/logs_20o20i_mixup_entro.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 50 --batch-size 64 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_40ood_20id_mixup_entro_IN32 --ood-ratio 0.4 --ind-ratio 0.2 --entro --mixup > logs/logs_40o20i_mixup_entro.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 50 --batch-size 64 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_60ood_20id_ce_entro_IN32 --ood-ratio 0.6 --ind-ratio 0.2 --entro > logs/logs_60o20i_ce_entro.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 50 --batch-size 64 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_40ood_40id_ce_entro_IN32 --ood-ratio 0.4 --ind-ratio 0.4 --entro > logs/logs_40o40i_ce_entro.txt

# DSOS on different corruptions of CIFAR-100 with ID bootstrapping only, OOD softening only, ID+OOD correction. Using batch size 32 is important to reach good accuracies.
# 20 ood 20 id
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 100 --batch-size 32 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_20ood_20id_DSOSboot_IN32 --ood-ratio 0.2 --ind-ratio 0.2 --track --dsos --mixup --boot --entro --resume checkpoints/preresnet18_cifar100/cifar100_20ood_20id_mixup_entro_IN32/1.0/preresnet18_cifar100_49.pth.tar > logs/logs_20o20i_DSOSboot.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 100 --batch-size 32 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_20ood_20id_DSOSsoft_IN32 --ood-ratio 0.2 --ind-ratio 0.2 --track --dsos --mixup --alpha .05 --soft --entro --resume checkpoints/preresnet18_cifar100/cifar100_20ood_20id_mixup_entro_IN32/1.0/preresnet18_cifar100_49.pth.tar > logs/logs_20o20i_DSOSsoft.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 100 --batch-size 32 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_20ood_20id_DSOS_IN32 --ood-ratio 0.2 --ind-ratio 0.2 --track --dsos --mixup --alpha .05 --soft --entro --boot --resume checkpoints/preresnet18_cifar100/cifar100_20ood_20id_mixup_entro_IN32/1.0/preresnet18_cifar100_49.pth.tar > logs/logs_20o20i_DSOS.txt

#40 ood 20 id
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 100 --batch-size 32 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_40ood_20id_DSOSboot_IN32 --ood-ratio 0.4 --ind-ratio 0.2 --track --dsos --mixup --boot --entro --resume checkpoints/preresnet18_cifar100/cifar100_40ood_20id_mixup_entro_IN32/1.0/preresnet18_cifar100_49.pth.tar > logs/logs_40o20i_DSOSboot.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 100 --batch-size 32 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_40ood_20id_DSOSsoft_IN32 --ood-ratio 0.4 --ind-ratio 0.2 --track --dsos --mixup --alpha .05 --soft --entro --resume checkpoints/preresnet18_cifar100/cifar100_40ood_20id_mixup_entro_IN32/1.0/preresnet18_cifar100_49.pth.tar > logs/logs_40o20i_DSOSsoft.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 100 --batch-size 32 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_40ood_20id_DSOS_IN32 --ood-ratio 0.4 --ind-ratio 0.2 --track --dsos --mixup --alpha .05 --soft --entro --boot --resume checkpoints/preresnet18_cifar100/cifar100_40ood_20id_mixup_entro_IN32/1.0/preresnet18_cifar100_49.pth.tar > logs/logs_40o20i_DSOS.txt

# 60 ood 20 id
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 100 --batch-size 32 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_60ood_20id_DSOSboot_IN32 --ood-ratio 0.6 --ind-ratio 0.2 --track --dsos --mixup --boot --entro --resume checkpoints/preresnet18_cifar100/cifar100_60ood_20id_ce_entro_IN32/1.0/preresnet18_cifar100_49.pth.tar > logs/logs_60o20i_DSOSboot.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 100 --batch-size 32 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_60ood_20id_DSOSsoft_IN32 --ood-ratio 0.6 --ind-ratio 0.2 --track --dsos --mixup --alpha .05 --soft --entro --resume checkpoints/preresnet18_cifar100/cifar100_60ood_20id_ce_entro_IN32/1.0/preresnet18_cifar100_49.pth.tar > logs/logs_60o20i_DSOSsoft.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 100 --batch-size 32 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_60ood_20id_DSOS_IN32 --ood-ratio 0.6 --ind-ratio 0.2 --track --dsos --mixup --alpha .05 --soft --entro --boot --resume checkpoints/preresnet18_cifar100/cifar100_60ood_20id_ce_entro_IN32/1.0/preresnet18_cifar100_49.pth.tar > logs/logs_60o20i_DSOS.txt

# 40 ood 40 id
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 100 --batch-size 32 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_40ood_40id_DSOSboot_IN32 --ood-ratio 0.4 --ind-ratio 0.4 --track --dsos --mixup --boot --entro --resume checkpoints/preresnet18_cifar100/cifar100_40ood_40id_ce_entro_IN32/1.0/preresnet18_cifar100_49.pth.tar > logs/logs_40o40i_DSOSboot.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 100 --batch-size 32 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_40ood_40id_DSOSsoft_IN32 --ood-ratio 0.4 --ind-ratio 0.4 --track --dsos --mixup --alpha .05 --soft --entro --resume checkpoints/preresnet18_cifar100/cifar100_40ood_40id_ce_entro_IN32/1.0/preresnet18_cifar100_49.pth.tar > logs/logs_40o40i_DSOSsoft.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 100 --batch-size 32 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_40ood_40id_DSOS_IN32 --ood-ratio 0.4 --ind-ratio 0.4 --track --dsos --mixup --alpha .05 --soft --entro --boot --resume checkpoints/preresnet18_cifar100/cifar100_40ood_40id_ce_entro_IN32/1.0/preresnet18_cifar100_49.pth.tar > logs/logs_40o40i_DSOS.txt

# CIFAR-100 Ablation study
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 100 --batch-size 32 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_20ood_20id_DSOS_IN32 --ood-ratio 0.2 --ind-ratio 0.2 --mixup --entro --resume checkpoints/preresnet18_cifar100/cifar100_20ood_20id_mixup_entro_IN32/1.0/preresnet18_cifar100_49.pth.tar > logs/logs_20o20i_abla_mixentro.txt

CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 100 --batch-size 32 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_20ood_20id_DSOS_IN32 --ood-ratio 0.2 --ind-ratio 0.2 --track --mixup --entro --resume checkpoints/preresnet18_cifar100/cifar100_20ood_20id_mixup_entro_IN32/1.0/preresnet18_cifar100_49.pth.tar > logs/logs_20o20i_abla_mixentroBN.txt

CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --epochs 100 --batch-size 32 --net preresnet18 --lr 0.03 --steps 50 80 --seed 1 --exp-name cifar100_20ood_20id_DSOS_IN32 --ood-ratio 0.2 --ind-ratio 0.2 --track --mixup --dsos --boot --entro --resume checkpoints/preresnet18_cifar100/cifar100_20ood_20id_mixup_entro_IN32/1.0/preresnet18_cifar100_49.pth.tar > logs/logs_20o20i_abla_mixentroBNBoot.txt

#Webvision
CUDA_VISIBLE_DEVICES=0 python main.py --dataset webvision --epochs 100 --batch-size 32 --net inception --lr 0.01 --steps 50 80 --seed 2 --exp-name webvis_DSOS --track --dsos --mixup --alpha .05 --soft --entro --boot > logs/logs_webvis_DSOS.txt

#Miniimagenet

CUDA_VISIBLE_DEVICES=0 python main.py --dataset miniimagenet_preset --epochs 200 --batch-size 32 --net inception --lr 0.01 --steps 100 160 --seed 1 --exp-name miniimagenet_DSOS_0 --noise-ratio 0.0 --soft --entro --boot --alpha .05 --entro --dsos --mixup --track > logs/logs_mini_0.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset miniimagenet_preset --epochs 200 --batch-size 32 --net inception --lr 0.01 --steps 100 160 --seed 1 --exp-name miniimagenet_DSOS_30 --noise-ratio 0.3 --soft --entro --boot --alpha .05 --entro --dsos --mixup --track > logs/logs_mini_30.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset miniimagenet_preset --epochs 200 --batch-size 32 --net inception --lr 0.01 --steps 100 160 --seed 1 --exp-name miniimagenet_DSOS_50 --noise-ratio 0.5 --soft --entro --boot --alpha .05 --entro --dsos --mixup --track > logs/logs_mini_50.txt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset miniimagenet_preset --epochs 200 --batch-size 32 --net inception --lr 0.01 --steps 100 160 --seed 1 --exp-name miniimagenet_DSOS_80 --noise-ratio 0.8 --soft --entro --boot --alpha .05 --entro --dsos --mixup --track > logs/logs_mini_80.txt

#Clothing1M
CUDA_VISIBLE_DEVICES=0 python main.py --dataset clothing --epochs 100 --batch-size 32 --net resnet50 --lr 0.002 --steps 50 80 --seed 1 --exp-name clothing_DSOS_1 --track --dsos --entro --mixup --alpha .05 --soft --boot --correct-ep 1 > logs/logs_clothing_DSOS.txt 
