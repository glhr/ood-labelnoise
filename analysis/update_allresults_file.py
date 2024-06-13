from results_utils import run, scores_stats
from data_beautification import CHECKPOINTS


if __name__ == "__main__":

    ood_dataset_selection = dict()
    for id_ds in CHECKPOINTS:
        ood_dataset_selection[id_ds] = ["imagenet6_test","food101","svhn","mnist","texture","stanfordproducts","eurosat"]
    ood_dataset_selection["fgvc-aircraft"] = ["fgvc-aircraft_OSR_easy", "fgvc-aircraft_OSR_hard", "fgvc-aircraft_OSR_medium"]
    ood_dataset_selection["fgvc-cub"] = ["fgvc-cub_OSR_easy", "fgvc-cub_OSR_hard", "fgvc-cub_OSR_medium"]

    exclude_methods = []

    archs_selection = ["resnet","cct","mlpmixer","resnet50"]

    exclude_groups = set(CHECKPOINTS.keys())
    exclude_groups.remove("fgvc-aircraft")
    exclude_groups.remove("fgvc-cub")
    exclude_groups = []

    n_bins = 10

    run(exclude_methods, archs_selection, exclude_groups, ood_dataset_selection,
        result_csv="result_csvs/noisy_openood_results_xp.csv")
    scores_stats(methods=["msp","mls","knn-train","rmds-train"])
