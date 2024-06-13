import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve


from scipy import interpolate

def fnr_at_tnr(preds,labels,pos_label=1,tnr=0.95):
    fprs, tprs, thresholds = roc_curve(labels, preds, pos_label=pos_label)
    tnrs = 1-fprs
    fnrs = 1-tprs
    if all(tnrs < tnr):
        return 0, None
    elif all(tnrs >= tnr):
        idxs = [i for i, x in enumerate(tnrs) if x >= tnr]
        selected_idx = np.argmin(fnrs[idxs])
        return fnrs[idxs][selected_idx], thresholds[idxs][selected_idx]

    thresh_intrp= interpolate.interp1d(tnrs,thresholds)
    thresh = thresh_intrp(tnr)

    fnr_interp = interpolate.interp1d(thresholds,fnrs)
    fnr95 = fnr_interp(thresh)

    # plt.plot(thresholds, tnr, label='TNR')
    # plt.plot(thresholds, fnr, label='FNR')
    # plt.axvline(x = thresh, color = 'black', label = '@TNR95')
    # plt.legend()
    # plt.show()

    return fnr95.item(), thresh.item()

def fpr_at_tpr(preds, labels, pos_label=1, tpr=0.95):
    fprs, tprs, thresholds = roc_curve(labels, preds, pos_label=pos_label)
    if all(tprs < tpr):
        # No threshold allows TPR >= 0.95
        return 0, None
    elif all(tprs >= tpr):
        # All thresholds allow TPR >= 0.95, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tprs) if x >= tpr]
        selected_idx = np.argmin(fprs[idxs])
        return fprs[idxs][selected_idx], thresholds[idxs][selected_idx]

    thresh_intrp= interpolate.interp1d(tprs,thresholds)
    thresh = thresh_intrp(tpr)

    fpr_interp = interpolate.interp1d(thresholds,fprs)
    fpr = fpr_interp(thresh)

    # plt.plot(thresholds, tnr, label='TNR')
    # plt.plot(thresholds, fnr, label='FNR')
    # plt.axvline(x = thresh, color = 'black', label = '@TNR95')
    # plt.legend()
    # plt.show()

    return fpr.item(), thresh.item()

def calc_standard_metrics(preds,labels,pos_label=1):
    metrics = dict()
    metrics["auroc"] = roc_auc_score(labels,preds)
    metrics["fpr@95tpr"], thresh_95tpr = fpr_at_tpr(preds,labels,pos_label=pos_label,tpr=0.95)
    metrics["fnr@95tnr"], thresh_95tnr = fnr_at_tnr(preds,labels,pos_label=pos_label,tnr=0.95)

    metrics["thresh_95tpr"] = thresh_95tpr
    metrics["thresh_95tnr"] = thresh_95tnr

    precision, recall, thresholds = precision_recall_curve(1-labels,-preds,pos_label=1)
    metrics["aupr-in"] = auc(recall, precision)
    precision, recall, thresholds = precision_recall_curve(labels,preds,pos_label=1)
    metrics["aupr-out"] = auc(recall, precision)
    return metrics


    
