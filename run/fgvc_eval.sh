methods=(rankfeat msp mls gradnorm)

for method in ${methods[@]}
do
    for dataset in fgvc-cub_clean fgvc-cub_symm0.1 fgvc-cub_symm0.2
    do
        for checkpoint in 'best.ckpt' 'last*.ckpt'
        do
            python scripts/eval_ood.py \
            --id-data ${dataset} \
            --root ./results/${dataset}_resnet50_base_e600_lr0.01_default \
            --postprocessor $method \
            --save-score --save-csv \
            --id_loader_split train \
            --checkpoint $checkpoint
        done
    done
done


for method in ${methods[@]}
do
    for dataset in fgvc-aircraft_clean fgvc-aircraft_symm0.1 fgvc-aircraft_symm0.2
    do
        for checkpoint in 'best.ckpt' 'last*.ckpt'
        do
            python scripts/eval_ood.py \
            --id-data ${dataset} \
            --root ./results/${dataset}_resnet50_base_e600_lr0.01_default \
            --postprocessor $method \
            --save-score --save-csv \
            --id_loader_split train \
            --checkpoint $checkpoint
        done
    done
done