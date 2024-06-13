num_gpus=1
seeds=(0 2 20)
datasets=(    clothing1M_clean # clean labels
                clothing1M_cleanval # real noisy labels
                clothing1M_cleanval_symm # synthetic uniform (SU) noisy labels
                clothing1M_cleanval_asymm # synthetic class-conditional (SCC) noisy labels
            )

# ResNet architecture
for seed in ${seeds[@]}
do
    for dataset in ${datasets[@]}
    do
        python main.py \
            --config configs/datasets/clothing1M/${dataset}.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/networks/resnet18_224x224.yml \
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
            --config configs/datasets/clothing1M/${dataset}.yml \
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
            --config configs/datasets/clothing1M/${dataset}.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/networks/cct_224x224.yml \
            configs/pipelines/train/baseline_cct_clothing.yml \
            --seed ${seed} \
            --num_gpus ${num_gpus} 
    done
done