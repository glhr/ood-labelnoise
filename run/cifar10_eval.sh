methods=(rankfeat msp mls gradnorm)

datasets=(  cifar10 # clean labels
            cifar10_noisy_worse cifar10_noisy_agg cifar10_noisy_random1 # real noisy labels
            cifar10_symm_worse cifar10_symm_agg cifar10_symm_random1 # synthetic uniform (SU) noisy labels
            cifar10_asymm_worse cifar10_asymm_agg cifar10_symm_random1 # synthetic class-conditional (SCC) noisy labels
            )

function run () {
  local dataset=$1
  local model=$2
  local method=$3
  local arch=$4
  local checkpoint=$5
  python scripts/eval_ood.py \
	   --id-data ${dataset} \
	   --root ./results/${dataset}_${model} \
	   --postprocessor $method \
	   --save-score --save-csv \
 	   --id_loader_split train \
	   --architecture ${arch} \
	   --checkpoint $checkpoint
}

for method in ${methods[@]}
do
    for checkpoint in best.ckpt last*.ckpt
    do
        for dataset in ${datasets[@]}
        do
            model=resnet18_32x32_base_e100_lr0.1_default
            arch=resnet
            run $dataset $model $method $arch $checkpoint

            model=mlpmixer_base_mlpmixer_e500_lr0.001_default
            arch=mlpmixer
            run $dataset $model $method $arch $checkpoint

            model=cct_7_3x1_32_base_cct_e300_lr0.00055_default
            arch=cct
            run $dataset $model $method $arch $checkpoint
        done
    done
done


