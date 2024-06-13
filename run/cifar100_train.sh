seeds=(0 2 20)
datasets=(  cifar100 cifar100_clean_coarse # clean labels
            cifar100_symm_fine cifar100_symm_coarse # synthetic uniform (SU) noisy labels
            cifar100_asymm_fine cifar100_asymm_coarse # synthetic class-conditional (SCC) noisy labels
            )  

# ResNet architecture
for seed in ${seeds[@]}
do
    for dataset in ${datasets[@]}
    do
        python main.py \
            --config configs/datasets/cifar100/${dataset}.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/networks/resnet18_32x32.yml \
            configs/pipelines/train/baseline.yml \
            --seed ${seed} \
            --num_gpus ${num_gpus}
    done
done

# MLPMixer architecture
for seed in ${seeds[@]}
do
    for dataset in ${datasets[@]}
    do
        python main.py \
            --config configs/datasets/cifar100/${dataset}.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/networks/mlpmixer.yml \
            configs/pipelines/train/baseline_mlpmixer.yml \
            --seed ${seed} \
            --num_gpus ${num_gpus} 
    done
done

# CCT architecture
for seed in ${seeds[@]}
do
    for dataset in ${datasets[@]}
    do
        python main.py \
            --config configs/datasets/cifar100/${dataset}.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/networks/cct_cifar.yml \
            configs/pipelines/train/baseline_cct_cifar100.yml \
            --seed ${seed} \
            --num_gpus ${num_gpus}
    done
done