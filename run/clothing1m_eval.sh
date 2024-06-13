methods=(rankfeat msp mls gradnorm)

datasets=(    clothing1M_clean # clean labels
                clothing1M_cleanval # real noisy labels
                clothing1M_cleanval_symm # synthetic uniform (SU) noisy labels
                clothing1M_cleanval_asymm # synthetic class-conditional (SCC) noisy labels
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
            model=resnet18_224x224_base_e100_lr0.1_default
            arch=resnet
            run $dataset $model $method $arch $checkpoint

            model=mlpmixer_base_mlpmixer_e500_lr0.001_default
            arch=mlpmixer
            run $dataset $model $method $arch $checkpoint

            model=cct_7_7x2_224_base_cct_e300_lr0.0005_default
            arch=cct
            run $dataset $model $method $arch $checkpoint
        done
    done
done


