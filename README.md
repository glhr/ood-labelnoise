<h2 align="center">[CVPR'24] <a href="https://glhr.github.io/OOD-LabelNoise/">A noisy elephant in the room:<br> Is your out-of-distribution detector robust to label noise?</a></h2>

  <p align="center">
    <a href="https://arxiv.org/abs/2404.01775"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2402.15509-b31b1b.svg"></a> 
  </p>

<br>

*noisy vs. clean training labels?*             |  *ID mistakes vs. OOD images?* | *difficulty of the OOD set?*
:-------------------------:|:-------------------------:|:-------------------------:
![](illustrations/auroc_per_dataset.png)  |  ![](illustrations/incorrect_vs_ood.png)  | ![](illustrations/osr_fgvc.png)


## 🔎 About

Considering how pervasive the problem of label noise is in real-world image classification datasets, its effect on OOD detection is crucial to study.
To address this gap, we systematically analyse the label noise robustness of a wide range of OOD detectors. Specifically:
1. We present the first study of post-hoc OOD detection in the presence of noisy classification labels, examining the performance of 20 state-of-the-art methods under different types and levels of label noise in the training data. Our study includes multiple classification architectures and datasets, ranging from the beloved CIFAR10 to the more difficult Clothing1M, and shows that even at a low noise rate, the label noise setting poses an interesting challenge for many methods.
2. We revisit the notion that OOD detection performance correlates with ID accuracy, examining when and why this relation holds. Robustness to inaccurate classification requires that OOD detectors effectively separate mistakes on ID data from OOD samples - yet most existing methods confound the two.




## 👩🏻‍🏫 Getting started

### What's in this repo?

* the [analysis](analysis) folder contains the scripts used to process and analyse results.
  * The [analysis/paper_figures.ipynb](analysis/paper_figures.ipynb) notebook is a good place to start. It reproduces all the visualizations and results in the paper, supplementary material and poster.
* the [run](run) folder contains bash scripts to train the base classifiers on different sets of (clean or noisy) labels (e.g. [run/cifar10_train.sh](run/cifar10_train.sh)), and then evaluate post-hoc OOD detectors (e.g. [run/cifar10_eval.sh](run/cifar10_eval.sh)). Training checkpoints and OOD detection results are saved in the [results](results) folder.

The rest of the repo follows the structure of [OpenOOD](https://github.com/Jingkang50/OpenOOD):

* [data/images_classic](data/images_classic) contains the raw ID & OOD datasets and annotations. See [data/README.md](data/README.md) for download instructions.

* [data/benchmark_imglist](data/benchmark_imglist) contains the list of images and corresponding label for each train, val, test and OOD set. For example, the training labels for CIFAR-10N-Agg (9.01% noise rate) can be found in [data/benchmark_imglist/train_cifar10n_agg.txt](data/benchmark_imglist/train_cifar10n_agg.txt) . We provide all the .txt files used in our experiments, as well as the scripts used to generate them.

  * for the code used to generate the clean & real noisy label sets, see the dataset-specific notebooks in the [data/images_classic](data/images_classic) folder (.e.g  [create_txt_files_cifar10.ipynb](data/images_classic/CIFAR-N/create_txt_files_cifar10.ipynb), [create_txt_files_clothing1m.ipynb](data/images_classic/clothing1M/create_txt_files_clothing1m.ipynb), [create_txt_files_cub.ipynb](data/images_classic/CUB_200_2011/create_txt_files_cub.ipynb) ...)
  * synthetic label sets are generated from the [data/benchmark_imglist/generate_synth_labels.ipynb](data/benchmark_imglist/generate_synth_labels.ipynb) notebook. 

### Conda environments

This code was tested on Ubuntu 18.04 + CUDA 11.3 & Ubuntu 20.04 + CUDA 12.5 with Python 3.11.3 + PyTorch 2.0.1. CUDA & PyTorch are only necessary for training classifiers and evaluating OOD detectors yourself. If you are only interested in reproducing the paper's tables & visualizations, you can install a minimal environment.

#### Minimal environment

```bash
conda create --name ood-labelnoise-viz python=3.11.3
conda activate ood-labelnoise-viz
pip install -r requirements_viz.txt
```

#### Full environment

```shell
conda create -n ood-labelnoise python=3.11.3
conda activate ood-labelnoise
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
conda install gcc_linux-64 gxx_linux-64
pip install Cython==3.0.2
pip install -r requirements_full.txt
```

## OOD detection methods

We benchmark the following 20 post-hoc OOD detection methods (listed in the order that they are presented in the paper). Their implementations are based on the [OpenOOD](https://github.com/Jingkang50/OpenOOD) benchmark, except for MDSEnsemble and GRAM which we modified to better align with the original papers.

| Name | Implementation | Paper |
|---|---|---|
| MSP |  [BasePostprocessor](openood/postprocessors/base_postprocessor.py) | [Hendrycks et al. 2017](https://openreview.net/forum?id=Hkg4TI9xl) |
| TempScaling | [TemperatureScalingPostprocessor](openood/postprocessors/temp_scaling_postprocessor.py) | [Guo et al. 2017](https://dl.acm.org/doi/10.5555/3305381.3305518) |
| ODIN | [ODINPostprocessor](openood/postprocessors/odin_postprocessor.py) | [Liang et al. 2018](https://openreview.net/forum?id=H1VGkIxRZ) |
| GEN | [GENPostprocessor](openood/postprocessors/gen_postprocessor.py) | [Liu et al. 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_GEN_Pushing_the_Limits_of_Softmax-Based_Out-of-Distribution_Detection_CVPR_2023_paper.html) |
| MLS | [MaxLogitPostprocessor](openood/postprocessors/maxlogit_postprocessor.py) | [Hendrycks et al. 2022](https://proceedings.mlr.press/v162/hendrycks22a.html) |
| EBO | [EBOPostprocessor](openood/postprocessors/ebo_postprocessor.py) | [Liu et al. 2020](https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html) |
| REACT | [ReactPostprocessor](openood/postprocessors/react_postprocessor.py) | [Sun et al. 2021](https://proceedings.neurips.cc/paper/2021/hash/01894d6f048493d2cacde3c579c315a3-Abstract.html) |
| RankFeat | [RankFeatPostprocessor](openood/postprocessors/rankfeat_postprocessor.py) | [Song et al. 2022](https://openreview.net/forum?id=-deKNiSOXLG) |
| DICE | [DICEPostprocessor](openood/postprocessors/dice_postprocessor.py) | [Sun et al. 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4405_ECCV_2022_paper.php) |
| ASH | [ASHPostprocessor](openood/postprocessors/ash_postprocessor.py) | [Djurisic et al. 2023](https://openreview.net/forum?id=ndYXTEL6cZz) |
| MDS | [MDSPostprocessor](openood/postprocessors/mds_postprocessor.py) | [Lee et al. 2018](https://papers.nips.cc/paper_files/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html) |
| MDSEnsemble | [MDSEnsemblePostprocessorMod](openood/postprocessors/mds_ensemble_mod_postprocessor.py)| [Lee et al. 2018](https://papers.nips.cc/paper_files/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html) |
| RMDS | [RMDSPostprocessor](openood/postprocessors/rmds_postprocessor.py) | [Ren et al. 2021](https://arxiv.org/abs/2106.09022) |
| KLM | [KLMatchingPostprocessor](openood/postprocessors/kl_matching_postprocessor.py) | [Hendrycks et al. 2022](https://proceedings.mlr.press/v162/hendrycks22a.html) |
| OpenMax | [OpenMax](openood/postprocessors/openmax_postprocessor.py) | [Bendale et al. 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Bendale_Towards_Open_Set_CVPR_2016_paper.pdf) |
| SHE | [SHEPostprocessor](openood/postprocessors/she_postprocessor.py) | [Zhang et al. 2023](https://openreview.net/forum?id=KkazG4lgKL) |
| GRAM | [GRAMPostprocessorMod](openood/postprocessors/gram_mod_postprocessor.py) | [Sastry et al. 2020](https://proceedings.mlr.press/v119/sastry20a.html) |
| KNN | [KNNPostprocessor](openood/postprocessors/knn_postprocessor.py) | [Sun et al. 2022](https://proceedings.mlr.press/v162/sun22d.html) |
| VIM | [VIMPostprocessor](openood/postprocessors/vim_postprocessor.py) | [Wang et al. 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_ViM_Out-of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.pdf) |
| GradNorm | [GradNormPostprocessor](openood/postprocessors/gradnorm_postprocessor.py) | [Huang et al. 2021](https://proceedings.neurips.cc/paper/2021/hash/063e26c670d07bb7c4d30e6fc69fe056-Abstract.html) |

## 📝 Updates
- June 13th 2024: Code repo released
- June 7th 2024: [Project page](https://glhr.github.io/OOD-LabelNoise/) released

## 📚 Citation

If you find our work useful, please cite:

```bibtex
@InProceedings{Humblot-Renaux_2024_CVPR,
    author={Humblot-Renaux, Galadrielle and Escalera, Sergio and Moeslund, Thomas B.},
    title={A Noisy Elephant in the Room: Is Your Out-of-Distribution Detector Robust to Label Noise?},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2024},
    pages={22626-22636}
}
```

## ✉️ Contact

If you have have any issues or doubts about the code, please create a Github issue. Otherwise, you can contact me at gegeh@create.aau.dk

## 🤝🏼 Acknowledgements
- Our codebase heavily builds on the [OpenOOD benchmark](https://github.com/Jingkang50/OpenOOD). We list our main changes in the paper's [supplementary material](https://arxiv.org/src/2404.01775v1/anc/supplementary.pdf). 
- Our benchmark includes the [CIFAR-N](https://github.com/UCSC-REAL/cifar-10-100n) and [Clothing1M](https://github.com/Cysu/noisy_label) datasets. These are highly valuable as they provide pairs of clean vs. real noisy labels. 
- We use the [deep-significance](https://github.com/Kaleidophon/deep-significance) implementation of the Almost Stochastic Order test in our experimental comparisons.
- We follow the training procedure and splits from the [Semantic Shift Benchmark](https://github.com/sgvaze/SSB) to evaluate fine-grained semantic shift detection.
- The Compact Transformer and MLPMixer model implementation and training hyper-parameters are based on the following repositories: [Compact-Transformers](https://github.com/SHI-Labs/Compact-Transformers) and [vision-transformers-cifar10](https://github.com/kentaroy47/vision-transformers-cifar10).
