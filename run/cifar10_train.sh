num_gpus=1
seeds=(0 2 20)
datasets=(  cifar10 # clean labels
            cifar10_noisy_worse cifar10_noisy_agg cifar10_noisy_random1 # real noisy labels
            cifar10_symm_worse cifar10_symm_agg cifar10_symm_random1 # synthetic uniform (SU) noisy labels
            cifar10_asymm_worse cifar10_asymm_agg cifar10_symm_random1 # synthetic class-conditional (SCC) noisy labels
            )

# ResNet architecture
for seed in ${seeds[@]}
do
    for dataset in ${datasets[@]}
    do
        python main.py \
            --config configs/datasets/cifar10/${dataset}.yml \
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
            --config configs/datasets/cifar10/${dataset}.yml \
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
            --config configs/datasets/cifar10/${dataset}.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/networks/cct_cifar.yml \
            configs/pipelines/train/baseline_cct_cifar10.yml \
            --seed ${seed} \
            --num_gpus ${num_gpus}
    done
done
