seeds=(0 2 20)

for seed in ${seeds[@]}
    do
    for dataset in fgvc-cub_clean fgvc-cub_symm0.1 fgvc-cub_symm0.2
    do
        python main.py \
            --config configs/datasets/fgvc-cub/${dataset}.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/networks/resnet50.yml \
            configs/pipelines/train/baseline_fgvc-cub.yml \
            --network.pretrained True \
            --network.image_size 448x448 \
            --seed ${seed}
    done
done

for seed in ${seeds[@]}
    do
    for dataset in fgvc-aircraft_clean fgvc-aircraft_symm0.1 fgvc-aircraft_symm0.2
    do
        python main.py \
            --config configs/datasets/fgvc-aircraft/${dataset}.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/networks/resnet50.yml \
            configs/pipelines/train/baseline_fgvc-aircraft.yml \
            --network.pretrained True \
            --network.image_size 448x448 \
            --seed ${seed}
    done
done