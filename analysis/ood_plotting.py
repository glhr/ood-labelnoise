import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from ood_metrics import calc_standard_metrics

plt.tight_layout(pad=0)

def plot_correct_incorrect_ood( correct_data,
                                incorrect_data,
                                ood_data,
                                model_name=None,
                                plot="hist",
                                save=False,
                                save_folder="ood_plots/",
                                plot_labels=True,
                                plot_threshs=True,
                                hist_bins=20,
                                show=False,
                                **kwargs):
    
    c = dict()

    c["name"] = model_name if model_name is not None else ""
    c["preds"] = np.concatenate([correct_data, incorrect_data, ood_data])
    c["labels"] = np.concatenate([np.zeros_like(correct_data),
                                  np.ones_like(incorrect_data),
                                  2*np.ones_like(ood_data)]).astype(int)
    
    id_data = np.concatenate([correct_data, incorrect_data])
    
    id_color = "violet"
    correct_color = "green"
    incorrect_color = "red"
    ood_color = "grey"
    
    if "hist" in plot:

        ## ID vs. OOD
        count, bins, ignored = plt.hist([id_data,ood_data], hist_bins, density=True,
                                        color=[id_color,ood_color], label=["ID", "OOD"], alpha=0.7, histtype='step',fill=True)
        
        plt.gcf().set_size_inches(3, 2)
        if plot_labels:
            plt.xlabel("OOD scores")
            plt.ylabel("Density")

            handles, labels = plt.gca().get_legend_handles_labels()
            order = [0,1]
            plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower center", mode="expand", borderaxespad=0, ncol=4 if plot_threshs else 2)
        ylim = plt.gca().get_ylim()
        xlim = plt.gca().get_xlim()
        plt.tight_layout(pad=0)
        if save:
            plt.savefig(f"{save_folder}/id_ood_histo-{model_name}.png")
            plt.savefig(f"{save_folder}/id_ood_histo-{model_name}.pdf")
        elif show: plt.show()
        plt.close()


        ## Correct vs. OOD
        count, bins, ignored = plt.hist([correct_data, ood_data], hist_bins, density=True,
                                        color=[correct_color,ood_color],
                                        label=["correct (ID)", "OOD"], alpha=0.7, histtype='step',fill=True)
        plt.gcf().set_size_inches(3, 2)
        if plot_labels:
            plt.xlabel("OOD scores")
            plt.ylabel("Density")

            handles, labels = plt.gca().get_legend_handles_labels()
            order = [0,1]
            plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower center", mode="expand", borderaxespad=0, ncol=4 if plot_threshs else 2)
        plt.gca().set_ylim(ylim)
        plt.gca().set_xlim(xlim)
        plt.tight_layout(pad=0)
        if save:
            plt.savefig(f"{save_folder}/correct_ood_histo-{model_name}.png")
            plt.savefig(f"{save_folder}/correct_ood_histo-{model_name}.pdf")
        elif show: plt.show()
        plt.close()

        ## Incorrect vs. OOD
        count, bins, ignored = plt.hist([incorrect_data, ood_data], hist_bins, density=True,
                                        color=[incorrect_color,ood_color], label=["incorrect (ID)", "OOD"], alpha=0.7, histtype='step',fill=True)

        plt.gcf().set_size_inches(3, 2)
        if plot_labels:
            plt.xlabel("OOD scores")
            plt.ylabel("Density")

            handles, labels = plt.gca().get_legend_handles_labels()
            order = [0,1]
            plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower center", mode="expand", borderaxespad=0, ncol=4 if plot_threshs else 2)
        #if plot_tight: 
        plt.gca().set_ylim(ylim)
        plt.gca().set_xlim(xlim)
        plt.tight_layout(pad=0)
        if save:
            plt.savefig(f"{save_folder}/incorrect_ood_histo-{model_name}.png")
            plt.savefig(f"{save_folder}/incorrect_ood_histo-{model_name}.pdf")
        elif show: plt.show()
        plt.close()


        

def plot_ood_scores(id_data,ood_data,model_name=None,
                    clip=True,
                    calc_metrics=True
                    ):

    if clip:
        ood_data = ood_data[ood_data<=1]
        ood_data = ood_data[ood_data>=0]
        id_data = id_data[id_data<=1]
        id_data = id_data[id_data>=0]
    
    c = dict()

    c["name"] = model_name if model_name is not None else ""
    c["preds"] = np.concatenate([id_data, ood_data])
    c["labels"] = np.concatenate([np.zeros_like(id_data),np.ones_like(ood_data)]).astype(int)
    
    if calc_metrics:
        c["metrics"] = calc_standard_metrics(c["preds"],c["labels"],pos_label=1)

    return c
