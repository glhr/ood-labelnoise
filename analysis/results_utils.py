import numpy as np
import os
import pandas as pd
import sklearn
from ood_plotting import *
from tqdm import tqdm

import pandas as pd
import traceback

import os.path
import scipy

from data_beautification import nice_method_names, CHECKPOINTS, ARCHS, NETWORKS, EPOCHS, TRAINERS, LRS, MARKS

RESULTS_PATH = "../results"



def get_noise_rates(path_clean_txt, path_noisy_txt):
    clean_labels = pd.read_csv(path_clean_txt, header=None, sep=' ')
    clean_labels.columns = ['image', 'label_clean']
    noisy_labels = pd.read_csv(path_noisy_txt, header=None, sep=' ')
    noisy_labels.columns = ['image', 'label_noisy']

    df = pd.merge(clean_labels, noisy_labels, on="image", how="inner")
    confusion_matrix = pd.crosstab(df['label_clean'], df['label_noisy'])

    noise_rate_overall = 1 - np.sum(df["label_clean"] == df["label_noisy"]) / len(df)

    mislabeling_rate = []
    for i in range(len(confusion_matrix)):
        mislabeling_rate.append(1 - confusion_matrix.iloc[i][i] / confusion_matrix.iloc[i].sum())
    mislabeling_rate_npy = np.array(mislabeling_rate)

    return {
        "noise_rate_overall": noise_rate_overall,
        "noise_rates_cls": mislabeling_rate,
        "noise_rate_cls_worst": np.max(mislabeling_rate_npy),
        "noise_rate_cls_best": np.min(mislabeling_rate_npy),
    }

split="train"
txt_root = "../data/benchmark_imglist/"

meta = {
    "cifar10": {
            "clean_txt": f"cifar10/{split}_cifar10.txt",
            "noisy_txt": f"cifar10/{split}_cifar10.txt",
            "nice_name": "CIFAR-10"
        },
    "cifar10_noisy_agg": {
        "clean_txt": f"cifar10/{split}_cifar10.txt",
        "noisy_txt": f"cifar10/{split}_cifar10n_agg.txt",
        "nice_name": "CIFAR-10-Agg-N"
    },
    "cifar10_noisy_random1": {
            "clean_txt": f"cifar10/{split}_cifar10.txt",
            "noisy_txt": f"cifar10/{split}_cifar10n_random1.txt",
            "nice_name": "CIFAR-10-Random1-N"
        },
    "cifar10_noisy_worse": {
        "clean_txt": f"cifar10/{split}_cifar10.txt",
        "noisy_txt": f"cifar10/{split}_cifar10n_worse.txt",
        "nice_name": "CIFAR-10-Worse-N"
    },
    "cifar10_symm_agg": {
            "clean_txt": f"cifar10/{split}_cifar10.txt",
            "noisy_txt": f"cifar10/{split}_cifar10symm_agg.txt",
            "nice_name": "CIFAR-10-Agg-SU"
        },
    "cifar10_symm_random1": {
            "clean_txt": f"cifar10/{split}_cifar10.txt",
            "noisy_txt": f"cifar10/{split}_cifar10symm_random1.txt",
            "nice_name": "CIFAR-10-Random1-SU"
        },
    "cifar10_symm_worse": {
        "clean_txt": f"cifar10/{split}_cifar10.txt",
        "noisy_txt": f"cifar10/{split}_cifar10symm_worse.txt",
        "nice_name": "CIFAR-10-Worse-SU"
    },
    "cifar10_asymm_agg": {
            "clean_txt": f"cifar10/{split}_cifar10.txt",
            "noisy_txt": f"cifar10/{split}_cifar10asymm_agg.txt",
            "nice_name": "CIFAR-10-Agg-SCC"
        },
    "cifar10_asymm_random1": {
            "clean_txt": f"cifar10/{split}_cifar10.txt",
            "noisy_txt": f"cifar10/{split}_cifar10asymm_random1.txt",
            "nice_name": "CIFAR-10-Random1-SCC"
        },
    "cifar10_asymm_worse": {
        "clean_txt": f"cifar10/{split}_cifar10.txt",
        "noisy_txt": f"cifar10/{split}_cifar10asymm_worse.txt",
        "nice_name": "CIFAR-10-Worse-SCC"
    },
    "cifar100": {
        "clean_txt": f"cifar100/{split}_cifar100.txt",
        "noisy_txt": f"cifar100/{split}_cifar100.txt",
        "nice_name": "CIFAR-100-Fine"
    },
    "cifar100_noisy_fine": {
        "clean_txt": f"cifar100/{split}_cifar100.txt",
        "noisy_txt": f"cifar100/{split}_cifar100n_noisyfine.txt",
        "nice_name": "CIFAR-100-Fine-N"
    },
    "cifar100_symm_fine": {
        "clean_txt": f"cifar100/{split}_cifar100.txt",
        "noisy_txt": f"cifar100/{split}_cifar100symm_noisyfine.txt",
        "nice_name": "CIFAR-100-Fine-SU"
    },
    "cifar100_asymm_fine": {
        "clean_txt": f"cifar100/{split}_cifar100.txt",
        "noisy_txt": f"cifar100/{split}_cifar100asymm_noisyfine.txt",
        "nice_name": "CIFAR-100-Fine-SCC"
    },
    "cifar100_clean_coarse": {
        "clean_txt": f"cifar100/{split}_cifar100n_cleancoarse.txt",
        "noisy_txt": f"cifar100/{split}_cifar100n_cleancoarse.txt",
        "nice_name": "CIFAR-100-Coarse"
    },
    "cifar100_noisy_coarse": {
        "clean_txt": f"cifar100/{split}_cifar100n_cleancoarse.txt",
        "noisy_txt": f"cifar100/{split}_cifar100n_noisycoarse.txt",
        "nice_name": "CIFAR-100-Coarse-N"
    },
    "cifar100_symm_coarse": {
        "clean_txt": f"cifar100/{split}_cifar100n_cleancoarse.txt",
        "noisy_txt": f"cifar100/{split}_cifar100symm_noisycoarse.txt",
        "nice_name": "CIFAR-100-Coarse-SU"
    },
    "cifar100_asymm_coarse": {
        "clean_txt": f"cifar100/{split}_cifar100n_cleancoarse.txt",
        "noisy_txt": f"cifar100/{split}_cifar100asymm_noisycoarse.txt",
        "nice_name": "CIFAR-100-Coarse-SCC"
    },
    "clothing1M_clean": {
        "clean_txt": f"clothing1M/{split}_clothing1M_clean.txt",
        "noisy_txt": f"clothing1M/{split}_clothing1M_clean.txt",
        "nice_name": "Clothing1M"
    },
    "clothing1M_cleanval": {
        "clean_txt": f"clothing1M/{split}_clothing1M_clean.txt",
        "noisy_txt": f"clothing1M/{split}_clothing1M_noisy.txt",
        "nice_name": "Clothing1M-N"
    },
    "clothing1M_cleanval_symm": {
        "clean_txt": f"clothing1M/{split}_clothing1M_clean.txt",
        "noisy_txt": f"clothing1M/{split}_clothing1M_symm.txt",
        "nice_name": "Clothing1M-SU"
    },
    "clothing1M_cleanval_asymm": {
        "clean_txt": f"clothing1M/{split}_clothing1M_clean.txt",
        "noisy_txt": f"clothing1M/{split}_clothing1M_asymm.txt",
        "nice_name": "Clothing1M-SCC"
    },
    "fgvc-aircraft_clean": {
        "clean_txt": f"fgvc-aircraft/{split}_fgvc-variant_clean.txt",
        "noisy_txt": f"fgvc-aircraft/{split}_fgvc-variant_clean.txt",
        "nice_name": "FGVC-Aircraft",
    },
    "fgvc-aircraft_symm0.1": {
        "clean_txt": f"fgvc-aircraft/{split}_fgvc-variant_clean.txt",
        "noisy_txt": f"fgvc-aircraft/{split}_fgvc-variant_symm0.1.txt",
        "nice_name": "FGVC-Aircraft-SU(0.1)",
    },
    "fgvc-aircraft_symm0.2": {
        "clean_txt": f"fgvc-aircraft/{split}_fgvc-variant_clean.txt",
        "noisy_txt": f"fgvc-aircraft/{split}_fgvc-variant_symm0.2.txt",
        "nice_name": "FGVC-Aircraft-SU(0.2)",
    },
    "fgvc-cub_clean": {
        "clean_txt": f"fgvc-cub/{split}_fgvc-cub_clean.txt",
        "noisy_txt": f"fgvc-cub/{split}_fgvc-cub_clean.txt",
        "nice_name": "CUB",
    },
    "fgvc-cub_symm0.1": {
        "clean_txt": f"fgvc-cub/{split}_fgvc-cub_clean.txt",
        "noisy_txt": f"fgvc-cub/{split}_fgvc-cub_symm0.1.txt",
        "nice_name": "CUB-SU(0.1)",
    },
    "fgvc-cub_symm0.2": {
        "clean_txt": f"fgvc-cub/{split}_fgvc-cub_clean.txt",
        "noisy_txt": f"fgvc-cub/{split}_fgvc-cub_symm0.2.txt",
        "nice_name": "CUB-SU(0.2)",
    },
}

for ds in meta:
    noise_dict = get_noise_rates(txt_root+meta[ds]["clean_txt"],txt_root+meta[ds]["noisy_txt"])
    meta[ds].update(noise_dict)

def get_model_checkpoints(IID_dataset,arch):
    try:
        id_ds = IID_dataset.split('_')[0]
        network = NETWORKS[arch][id_ds]
        epoch = EPOCHS[arch][id_ds]
        trainer = TRAINERS[arch][id_ds]
        lr = LRS[arch][id_ds]
        mark = MARKS[arch][id_ds]
        return {
            "knn-train":          f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "klm-train":          f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "ash":          f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "ash_s":          f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "react_mod-train":        f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "odin_mod":         f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "odin_nopert":         f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "odin_notemp":         f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "msp":          f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "mls":          f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "mds-val":          f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "mds-train":          f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "mds_ensemble_mod2-train": f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "mds_ensemble_mod2-val": f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "rankfeat":     f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "dice_mod-train":         f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "gram_mod-val":         f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "gram_mod-train":         f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "vim":          f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "rmds-val":         f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "rmds-train":         f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "she-val":          f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "she-train":          f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "temp_scaling-val": f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "temp_scaling-train": f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "gradnorm":     f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "openmax-val":      f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "openmax-train":      f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "gen":          f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "ebo":          f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
            "ebo_own":          f"_{network}_{trainer}_e{epoch}_lr{lr}_{mark}",
        }
    except Exception as e:
        #traceback.print_exc()
        return None


def get_res_from_scores(res_iid, res_ood, res_iid_correct=None, res_iid_incorrect=None,
                        model_name="", plot="", save_folder=None, save=True, normalize_scores=True,calc_metrics=True):

    if np.isnan(res_iid).any() or np.isnan(res_ood).any():
        print(f"(before norm) res_iid or res_ood contain nans {res_iid} {res_ood} ")
        return

    # normalize
    if normalize_scores:
        max_val = np.max(np.concatenate([res_iid,res_ood]))
        min_val = np.min(np.concatenate([res_iid,res_ood]))

        res_iid = 1-(res_iid-min_val) / (max_val-min_val)
        res_ood = 1-(res_ood-min_val) / (max_val-min_val)

    if np.isnan(res_iid).any() or np.isnan(res_ood).any():
        print(f"(before norm) res_iid or res_ood contain nans {res_iid} {res_ood} ")
        return

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)

    #print(save,model_name)

    if res_iid_correct is not None and res_iid_incorrect is not None:
        plot_correct_incorrect_ood(res_iid_correct,
                                    res_iid_incorrect,
                                    res_ood,
                                    plot=plot,
                                    hist_bins=50,
                                    model_name=model_name,
                                    save_folder=save_folder,
                                    save=save)

    return plot_ood_scores(res_iid, res_ood,
                            clip=False,
                            calc_metrics=calc_metrics,
                            model_name=model_name)

def run(exclude_methods, archs_selection, exclude_groups, ood_dataset_selection, 
        result_csv="result_csvs/noisy_openood_results_xp.csv"):
    metrics_list = []
    method_selection = set()
    overwrite = True

    def get_result_str_from_dict(res_dict):
        res_string = f'{res_dict["iid_dataset"]}-{res_dict["training_dataset"]}_{res_dict["arch"]}_{res_dict["seed"]}_{res_dict["checkpoint"]}_{res_dict["method"]}'
        return res_string

    if not overwrite:
        try:
            df_old_results = pd.read_csv(result_csv)
            metrics_list = df_old_results.to_dict('records')
            existing_set = set(
                [get_result_str_from_dict(d) for d in metrics_list]
            )
        except:
            print(f"Couldn't load existing results file, starting from scratch")
            metrics_list = []
            existing_set = set()
    else:
        metrics_list = []
        existing_set = set()

    print(CHECKPOINTS.keys())
    for id_dataset_group in CHECKPOINTS.keys():
        if id_dataset_group in exclude_groups:
            print(f"skipping {id_dataset_group}")
            continue
        print(f"Processing {id_dataset_group}")
        for dataset in CHECKPOINTS[id_dataset_group]:
            for arch in archs_selection:
                model_checkpoints = get_model_checkpoints(dataset,arch)
                if model_checkpoints is None:
                    #print(f"model checkpoints is None")
                    continue
                for method, suffix in tqdm(model_checkpoints.items()):
                    if method in exclude_methods: continue
                    if method not in nice_method_names: continue

                    method_selection.add(method)
                    path = f"{dataset}{suffix}"

                    pkl_file_name = method

                    # skip results computation if we already have them
                    try:
                        dataset_meta = meta[dataset]
                    except Exception as e:
                        print(e)
                        continue

                    found_all_results=True
                    for seed in ["s0","s2","s20"]:

                        for scores_folder in ["best.ckpt","last*.ckpt"]:
                            res_dict = {
                                    "arch": arch,
                                    "method": method,
                                    "training_dataset": dataset,
                                    "iid_dataset": id_dataset_group,
                                    "class": "all",
                                    "seed": seed,
                                    "checkpoint": scores_folder
                                }
                            if get_result_str_from_dict(res_dict) in existing_set:
                                continue
                            pkl_data = None
                            try:
                                pkl_file = f"{RESULTS_PATH}/{path}/{seed}/scores-{scores_folder}/{pkl_file_name}.pkl"
                                pkl_data = np.load(pkl_file, allow_pickle=True)
                            except Exception as e:
                                found_all_results = False

                            if pkl_data is None or not found_all_results :
                                continue

                            print(f"Processing {pkl_file}\r", end='',flush=True)

                            id_preds = pkl_data["id_preds"]
                            id_labels = pkl_data["id_labels"]
                            is_correct = (id_preds == id_labels).int()
                            accuracy = is_correct.float().mean()
                            num_classes = len(id_labels.unique())

                            res_iid_overall = pkl_data["id"]["test"][1].flatten()

                            res_iid_correct = res_iid_overall[is_correct.bool()]
                            res_iid_incorrect = res_iid_overall[~is_correct.bool()]

                            y_true, y_prob = is_correct,res_iid_overall

                            failure_auroc = sklearn.metrics.roc_auc_score(is_correct,res_iid_overall)

                            ood_datasets_near = {
                                k:"near" for k in pkl_data["ood"]["near"].keys()
                            }
                            ood_datasets_far = {
                                k:"far" for k in pkl_data["ood"]["far"].keys()
                            }
                            ood_datasets = {**ood_datasets_near, **ood_datasets_far}

                            for ood_dataset,group in ood_datasets.items():
                                if not ood_dataset in ood_dataset_selection[id_dataset_group]:
                                    print(f"Skipping {ood_dataset}")
                                    continue

                                res_ood = pkl_data["ood"][group][ood_dataset][1].flatten()

                                model_name = f"{method}-{arch}-OOD{ood_dataset}-{dataset}-seed{seed}"

                                res = get_res_from_scores(res_iid_overall, res_ood,
                                                            res_iid_correct=res_iid_correct,
                                                            res_iid_incorrect=res_iid_incorrect,
                                                            model_name=model_name,
                                                            plot="",save_folder=f"_ood_plots/{method}")

                                id_incorrect_probs = y_prob[~is_correct.bool()]
                                incorrect_vs_ood_labels = np.concatenate([np.zeros_like(id_incorrect_probs),np.ones_like(res_ood)])
                                incorrect_vs_ood_probs = np.concatenate([-id_incorrect_probs,-res_ood])
                                incorrect_ood_auroc = sklearn.metrics.roc_auc_score(incorrect_vs_ood_labels, incorrect_vs_ood_probs)

                                id_correct_probs = y_prob[is_correct.bool()]
                                correct_vs_ood_labels = np.concatenate([np.zeros_like(id_correct_probs),np.ones_like(res_ood)])
                                correct_vs_ood_probs = np.concatenate([-id_correct_probs,-res_ood])
                                correct_ood_auroc = sklearn.metrics.roc_auc_score(correct_vs_ood_labels, correct_vs_ood_probs)

                                metrics_dict_overall = {
                                    "arch": arch,
                                    "method": method,
                                    "training_dataset": dataset,
                                    "iid_dataset": id_dataset_group,
                                    "num_classes": num_classes,
                                    "accuracy": accuracy.item(),
                                    "failure_auroc": failure_auroc,
                                    "incorrect_ood_auroc": incorrect_ood_auroc,
                                    "correct_ood_auroc": correct_ood_auroc,
                                    "noise_rate": dataset_meta["noise_rate_overall"],
                                    "seed": seed,
                                    "checkpoint": scores_folder,
                                    "ood_dataset": ood_dataset,
                                    "auroc": res["metrics"]["auroc"]
                                }
                                metrics_list.append(metrics_dict_overall)
                                

    df = pd.DataFrame(metrics_list)
    df.to_csv(result_csv, index=False)

def visualize_scores(method,arch,checkpoint,dataset,ood_datasets_to_plot,seed,meta,save_plot=False):
    scores_folder=checkpoint
    pkl_file_name = method
    model_checkpoints = get_model_checkpoints(dataset,arch)
    suffix = model_checkpoints[method]

    #print(dataset)

    path = f"{dataset}{suffix}"

    res_iid_overall = np.array([])
    res_ood_overall = np.array([])

    pkl_file = f"{RESULTS_PATH}/{path}/{seed}/scores-{scores_folder}/{pkl_file_name}.pkl"

    pkl_data = np.load(pkl_file, allow_pickle=True)
    #print(pkl_data)
    #raise Exception

    res_iid = pkl_data["id"]["test"][1].flatten()
    id_preds = pkl_data["id_preds"]
    id_labels = pkl_data["id_labels"]
    is_correct = (id_preds == id_labels).int()

    current_max = np.max(res_iid)
    current_min = np.min(res_iid)

    res_iid_overall = res_iid
    res_iid_correct = res_iid_overall[is_correct.bool()]
    res_iid_incorrect = res_iid_overall[~is_correct.bool()]

    id_iqr = scipy.stats.iqr(res_iid_overall)
    id_median = np.median(res_iid_overall)

    ood_datasets_near = {
        k:"near" for k in pkl_data["ood"]["near"].keys()
    }
    ood_datasets_far = {
        k:"far" for k in pkl_data["ood"]["far"].keys()
    }
    ood_datasets = {**ood_datasets_near, **ood_datasets_far}

    #res_ood_overall =
    list_of_dicts = []
    for ood_dataset,group in ood_datasets.items():

        res_ood = pkl_data["ood"][group][ood_dataset][1].flatten()

        ood_max = np.max(res_ood)
        ood_min = np.min(res_ood)
        current_max = np.max([current_max,ood_max])
        current_min = np.min([current_min,ood_min])
        res_ood_overall = res_ood

        ood_median = np.median(res_ood_overall)
        ood_iqr = scipy.stats.iqr(res_ood_overall)

        model_name = f"{method}-{arch}-OOD{ood_dataset}-{dataset}-seed{seed}-{checkpoint}"
        plot="hist"
        res = get_res_from_scores(res_iid_overall, res_ood_overall,
                                    res_iid_correct=res_iid_correct,
                                    res_iid_incorrect=res_iid_incorrect,
                                    model_name=model_name,plot=plot,
                                    save_folder=f"ood_plots/{method}",save=save_plot and ood_dataset in ood_datasets_to_plot,
                                    normalize_scores=False, calc_metrics=False)
        #print(res)
        noise_rate = meta[dataset]["noise_rate_overall"]

        list_of_dicts.append({
            "dataset": dataset,
            "checkpoint": checkpoint,
            "id_median": id_median,
            "ood_median": ood_median,
            "id_iqr": id_iqr,
            "ood_iqr": ood_iqr,
            "noise_rate": noise_rate,
            "ood_dataset": ood_dataset,
            "seed": seed
        })

    for d in list_of_dicts:
        d["scores_max"] = current_max
        d["scores_min"] = current_min
    return list_of_dicts


def scores_stats(methods=["msp","mls","knn-train"],
                 seeds_to_plot=["s0"],
                 archs_to_plot=["resnet"],
                 checkpoints_to_plot=["last*.ckpt","best.ckpt"],
                 datasets_to_plot=["cifar10","cifar10_noisy_agg","cifar10_symm_agg","cifar10_asymm_agg",
                                   "cifar10_noisy_random1","cifar10_symm_random1","cifar10_asymm_random1",
                                   "cifar10_noisy_worse","cifar10_symm_worse","cifar10_asymm_worse"],
                 ood_datasets_to_plot=["svhn"]):

    overwrite_methods = methods
    for method in methods:
        fname = f"./result_csvs/scores-stats_{method}.csv"
        if os.path.isfile(fname) and not method in overwrite_methods:
            print(f"Skipping {method}, already exists at {fname}")
            continue
        list_of_dicts = []
        print(method)
        for id_dataset in tqdm(["cifar10","cifar10_noisy_agg","cifar10_symm_agg","cifar10_asymm_agg",
                           "cifar10_noisy_random1","cifar10_symm_random1","cifar10_asymm_random1",
                           "cifar10_noisy_worse","cifar10_symm_worse","cifar10_asymm_worse",
                           "cifar100","cifar100_noisy_fine","cifar100_symm_fine","cifar100_asymm_fine",
                           "cifar100_clean_coarse","cifar100_noisy_coarse","cifar100_symm_coarse","cifar100_asymm_coarse",
                           "clothing1M_clean","clothing1M_cleanval","clothing1M_cleanval_symm","clothing1M_cleanval_asymm"]):
            for arch in ["resnet","cct","mlpmixer","resnet50"]:
                for checkpoint in ["last*.ckpt","best.ckpt"]:
                    for seed in ["s0","s2","s20"]:
                        try:
                            save_plot = (id_dataset in datasets_to_plot and
                                    arch in archs_to_plot and
                                    checkpoint in checkpoints_to_plot and
                                    seed in seeds_to_plot)
                            
                            list_of_dicts.extend(visualize_scores(method, arch, checkpoint, id_dataset, ood_datasets_to_plot, seed, meta, save_plot=save_plot))
                        except Exception as e:
                            pass
                            #print(arch,checkpoint,seed)
                            #traceback.print_exc()

        df_scores = pd.DataFrame(list_of_dicts)
        df_scores.to_csv(fname)

